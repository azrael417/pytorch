#include <torch/optim/adam.h>

#include <torch/csrc/autograd/variable.h>
#include <torch/nn/module.h>
#include <torch/serialize/archive.h>
#include <torch/utils.h>

#include <ATen/ATen.h>
#include <c10/util/irange.h>

#include <cmath>
#include <functional>

// fused optimizers
#include <ATen/native/ForeachUtils.h>
#include <ATen/native/FusedAdam.h>
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_fused_adam.h>
#include <ATen/ops/_fused_adam_native.h>
#endif
#include <ATen/native/cuda/fused_adam_impl.cuh>

namespace torch {
namespace optim {

AdamOptions::AdamOptions(double lr) : lr_(lr) {}

bool operator==(const AdamOptions& lhs, const AdamOptions& rhs) {
  return (lhs.lr() == rhs.lr()) &&
      (std::get<0>(lhs.betas()) == std::get<0>(rhs.betas())) &&
      (std::get<1>(lhs.betas()) == std::get<1>(rhs.betas())) &&
      (lhs.eps() == rhs.eps()) &&
      (lhs.weight_decay() == rhs.weight_decay() &&
       (lhs.amsgrad() == rhs.amsgrad()));
}

void AdamOptions::serialize(torch::serialize::OutputArchive& archive) const {
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(lr);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(betas);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(eps);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(weight_decay);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(amsgrad);
}

void AdamOptions::serialize(torch::serialize::InputArchive& archive) {
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, lr);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(betas_t, betas);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, eps);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, weight_decay);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(bool, amsgrad);
}

double AdamOptions::get_lr() const {
  return lr();
}

void AdamOptions::set_lr(const double lr) {
  this->lr(lr);
}

bool operator==(const AdamParamState& lhs, const AdamParamState& rhs) {
  return (lhs.step() == rhs.step()) &&
      torch::equal(lhs.exp_avg(), rhs.exp_avg()) &&
      torch::equal(lhs.exp_avg_sq(), rhs.exp_avg_sq()) &&
      torch::equal_if_defined(lhs.max_exp_avg_sq(), rhs.max_exp_avg_sq());
}

void AdamParamState::serialize(torch::serialize::OutputArchive& archive) const {
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(step);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(exp_avg);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(exp_avg_sq);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(max_exp_avg_sq);
}

void AdamParamState::serialize(torch::serialize::InputArchive& archive) {
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(int64_t, step);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(Tensor, exp_avg);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(Tensor, exp_avg_sq);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(Tensor, max_exp_avg_sq);
}

bool Adam::_init_group(OptimizerParamGroup& group,
		       std::vector<std::optional<Tensor>>& params_with_grads,
		       std::vector<std::optional<Tensor>>& grads,
		       std::vector<std::optional<Tensor>>& exp_avgs,
		       std::vector<std::optional<Tensor>>& exp_avg_sqs,
		       std::vector<std::optional<Tensor>>& max_exp_avg_sqs,
		       std::vector<std::optional<Tensor>>& state_steps) {

  bool has_complex = false;
  for (auto& p : group.params()) {
    if (!p.grad().defined()) {
      continue;
    }
    has_complex |= p.is_complex();

    params_with_grads.push_back(p);
    TORCH_CHECK(!p.grad().is_sparse(), "Adam does not support sparse gradients" /*, please consider SparseAdam instead*/);
    grads.push_back(p.grad());

    auto param_state = state_.find(p.unsafeGetTensorImpl());
    auto& options = static_cast<AdamOptions&>(group.options());

    // Lazy State initialization
    if (param_state == state_.end()) {
      auto state = std::make_unique<AdamParamState>();

      if (options.fused()) {
	// check if device and datatype are supported
	_device_dtype_check_for_fused(p);
      }
      state->step(0);
      // Exponential moving average of gradient values
      state->exp_avg(torch::zeros_like(p, MemoryFormat::Preserve));
      // Exponential moving average of squared gradient values
      state->exp_avg_sq(torch::zeros_like(p, MemoryFormat::Preserve));
      if (options.amsgrad()) {
        // Maintains max of all exp. moving avg. of sq. grad. values
        state->max_exp_avg_sq(torch::zeros_like(p, MemoryFormat::Preserve));
      }
      state_[p.unsafeGetTensorImpl()] = std::move(state);
    }

    auto& state =
      static_cast<AdamParamState&>(*state_[p.unsafeGetTensorImpl()]);

    exp_avgs.push_back(state.exp_avg());
    exp_avg_sqs.push_back(state.exp_avg_sq());

     if (options.amsgrad()) {
       max_exp_avg_sqs.push_back(state.max_exp_avg_sq());
     }

     torch::Tensor steptens;
     if (options.fused()) {
       steptens = torch::tensor({state.step()}, TensorOptions().device(p.device()).dtype(torch::kFloat32));
     } else {
       steptens = torch::tensor({state.step()}, TensorOptions().dtype(torch::kLong));
     }
     state_steps.push_back(steptens);
  }

  return has_complex;
}
  
void _single_tensor_adam(std::vector<std::optional<Tensor>>& params_with_grad,
			 std::vector<std::optional<Tensor>>& grads,
			 std::vector<std::optional<Tensor>>& exp_avgs,
			 std::vector<std::optional<Tensor>>& exp_avg_sqs,
			 std::vector<std::optional<Tensor>>& max_exp_avg_sqs,
			 std::vector<std::optional<Tensor>>& state_steps,
			 bool amsgrad,
			 bool has_complex,
			 double beta1,
			 double beta2,
			 double lr,
			 double weight_decay,
			 double eps) {

  for(size_t i=0; i<params_with_grad.size(); ++i) {
    auto p = params_with_grad[i].value();
    auto grad = grads[i].value();
    auto& exp_avg = exp_avgs[i].value();
    auto& exp_avg_sq = exp_avg_sqs[i].value();
    auto& state_step = state_steps[i].value();

    // increment state counter
    state_step.add_(1);
    
    auto bias_correction1 = 1 - std::pow(beta1, state_step.item<long>());
    auto bias_correction2 = 1 - std::pow(beta2, state_step.item<long>());
    
    if (weight_decay != 0) {
      grad = grad.add(p, weight_decay);
    }
    
    // Decay the first and second moment running average coefficient
    exp_avg.mul_(beta1).add_(grad, 1 - beta1);
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, 1 - beta2);
    
    Tensor denom;
    if (amsgrad) {
      // Maintains the maximum of all 2nd moment running avg. till now
      auto& max_exp_avg_sq = max_exp_avg_sqs[i].value();
      torch::max_out(max_exp_avg_sq, exp_avg_sq, max_exp_avg_sq);
      // Use the max. for normalizing running avg. of gradient
      denom = (max_exp_avg_sq.sqrt() / sqrt(bias_correction2)).add_(eps);
    } else {
      denom =
	(exp_avg_sq.sqrt() / sqrt(bias_correction2)).add_(eps);
    }
    
    auto step_size = lr / bias_correction1;
    p.addcdiv_(exp_avg, denom, -step_size);
  }
}      


void _fused_tensor_adam(std::vector<std::optional<Tensor>>& params,
                        std::vector<std::optional<Tensor>>& grads,
                        std::vector<std::optional<Tensor>>& exp_avgs,
                        std::vector<std::optional<Tensor>>& exp_avg_sqs,
                        std::vector<std::optional<Tensor>>& max_exp_avg_sqs,
                        std::vector<std::optional<Tensor>>& state_steps,
                        bool amsgrad,
                        bool has_complex,
                        double beta1,
                        double beta2,
                        double lr,
                        double weight_decay,
                        double eps) {
  if(params.size() == 0) return;

  std::vector<std::vector<std::optional<Tensor>>> tensorlistlist{params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps};
  auto grouped_tensors = at::native::_group_tensors_by_first_tensors_device_and_dtype(tensorlistlist, false);
  for (auto& [key, value]: grouped_tensors) {
    auto device_tensorlistlist = std::get<0>(value);
    auto device = std::get<0>(key);
    auto device_params = cast_to_tensorlist(device_tensorlistlist[0]);
    auto device_grads = cast_to_tensorlist(device_tensorlistlist[1]);
    auto device_exp_avgs = cast_to_tensorlist(device_tensorlistlist[2]);
    auto device_exp_avg_sqs = cast_to_tensorlist(device_tensorlistlist[3]);
    auto device_max_exp_avg_sqs = cast_to_tensorlist(device_tensorlistlist[4]);
    auto device_state_steps = cast_to_tensorlist(device_tensorlistlist[5]);

    if(has_complex) {
      _view_as_real(device_params, device_grads, device_exp_avgs, device_exp_avg_sqs);
      if(amsgrad) {
	_view_as_real(device_params, device_grads, device_exp_avgs, device_exp_avg_sqs, device_max_exp_avg_sqs);
      }
    }

    // increase state steps:
    for(auto& device_state_step: device_state_steps) {
      device_state_step.add_(1);
    }
    auto lr_tensor = torch::tensor({lr}, TensorOptions().device(device).dtype(torch::kDouble));

    auto devname = c10::DeviceTypeName(device.type(), true);
    if (devname == "cuda") {
      at::native::_fused_adam_cuda_impl_(
					 at::TensorList(device_params),
					 at::TensorList(device_grads),
					 at::TensorList(device_exp_avgs),
					 at::TensorList(device_exp_avg_sqs),
					 at::TensorList(device_state_steps),
					 lr_tensor,
					 beta1,
					 beta2,
					 weight_decay,
					 eps,
					 false,
					 {},
					 {});
    } else if (devname == "cpu") {
      at::native::_fused_adam_kernel_cpu_(
					  at::TensorList(device_params),
					  at::TensorList(device_grads),
					  at::TensorList(device_exp_avgs),
					  at::TensorList(device_exp_avg_sqs),
					  at::TensorList(device_max_exp_avg_sqs),
					  at::TensorList(device_state_steps),
					  lr,
					  beta1,
					  beta2,
					  weight_decay,
					  eps,
					  false,
					  {},
					  {});
    } else {
      TORCH_CHECK(false, "Adam does not support fusing on device " + devname + " yet");
    }
  }
}
  
  
Tensor Adam::step(LossClosure closure) {
  NoGradGuard no_grad;
  Tensor loss = {};
  if (closure != nullptr) {
    at::AutoGradMode enable_grad(true);
    loss = closure();
  }

  std::cout << "PERFORMING A STEP WITH MY TWEAKED ADAM" << std::endl;
  
  for (auto& group : param_groups_) {

    // init the group
    std::vector<std::optional<Tensor>> params_with_grad, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps;
    auto& options = static_cast<AdamOptions&>(group.options());
    auto beta1 = std::get<0>(options.betas());
    auto beta2 = std::get<1>(options.betas());
    
    bool has_complex = _init_group(group, params_with_grad, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps);
    
    if (!options.fused()) {
      _single_tensor_adam(params_with_grad, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps,
			  options.amsgrad(), has_complex, beta1, beta2, options.lr(), options.weight_decay(), options.eps());
    } else {
      _fused_tensor_adam(params_with_grad, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps,
			 options.amsgrad(), has_complex, beta1, beta2, options.lr(), options.weight_decay(), options.eps());
    }
  }
  return loss;
}

void Adam::save(serialize::OutputArchive& archive) const {
  serialize(*this, archive);
}

void Adam::load(serialize::InputArchive& archive) {
  IValue pytorch_version;
  if (archive.try_read("pytorch_version", pytorch_version)) {
    serialize(*this, archive);
  } else { // deserializing archives saved in old format (prior to
           // version 1.5.0)
    TORCH_WARN(
        "Your serialized Adam optimizer is still using the old serialization format. "
        "You should re-save your Adam optimizer to use the new serialization format.");
    std::vector<int64_t> step_buffers;
    std::vector<at::Tensor> exp_average_buffers;
    std::vector<at::Tensor> exp_average_sq_buffers;
    std::vector<at::Tensor> max_exp_average_sq_buffers;
    torch::optim::serialize(archive, "step_buffers", step_buffers);
    torch::optim::serialize(
        archive, "exp_average_buffers", exp_average_buffers);
    torch::optim::serialize(
        archive, "exp_average_sq_buffers", exp_average_sq_buffers);
    torch::optim::serialize(
        archive, "max_exp_average_sq_buffers", max_exp_average_sq_buffers);
    // since there were no param_groups prior to version 1.5.0, assuming all
    // tensors are now in one param_group
    std::vector<Tensor> params = param_groups_.at(0).params();
    for (const auto idx : c10::irange(step_buffers.size())) {
      auto state = std::make_unique<AdamParamState>();
      state->step(step_buffers.at(idx));
      state->exp_avg(exp_average_buffers.at(idx));
      state->exp_avg_sq(exp_average_sq_buffers.at(idx));
      if (idx < max_exp_average_sq_buffers.size()) {
        state->max_exp_avg_sq(max_exp_average_sq_buffers.at(idx));
      }
      state_[params.at(idx).unsafeGetTensorImpl()] = std::move(state);
    }
  }
}
} // namespace optim
} // namespace torch
