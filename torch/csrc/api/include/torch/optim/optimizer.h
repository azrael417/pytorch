#pragma once

#include <ATen/Tensor.h>
#include <c10/util/Exception.h>
#include <c10/util/flat_hash_map.h>

#include <torch/arg.h>
#include <torch/csrc/Export.h>

#include <algorithm>
#include <functional>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/view_as_real.h>
#endif

// Forward declarations confuse Doxygen
#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace at {
class Tensor;
} // namespace at

namespace torch {
using at::Tensor;
namespace serialize {
class OutputArchive;
class InputArchive;
} // namespace serialize
} // namespace torch
#endif // DOXYGEN_SHOULD_SKIP_THIS

namespace torch {
namespace optim {

void _device_dtype_check_for_fused(const Tensor& param, bool cuda_unsupported = false) {
  std::vector<std::string> supported_devices = {"mps", "xpu", "cpu"};
  if (!cuda_unsupported) {
    supported_devices.push_back("cuda");
  }

  bool supported = param.is_floating_point();
  auto pdevice = c10::DeviceTypeName(param.device().type(), true);
  auto has_device = [&pdevice](std::string dev) { return dev == pdevice; };
  supported |= (std::find_if(supported_devices.begin(), supported_devices.end(), has_device) != supported_devices.end());

  TORCH_CHECK(!supported, "`fused=True` requires all the params to be floating point Tensors of supported devices");
}

template<typename... StatesAndGrads>
void _view_as_real(std::vector<Tensor>& params, StatesAndGrads&&... states_and_grads) {
  size_t pcount = params.size();
  for(size_t i=0; i<pcount; ++i) {
    if (params[i].is_complex()) {
      params[i] = at::view_as_real(params[i]);
      //for (auto& state: {std::forward<StatesAndGrads>(states_and_grads)...}) {
      //state[i] = at::view_as_real(state[i]);
      //}
      (..., (states_and_grads[i] = at::view_as_real(states_and_grads[i])));
    }
  }
}

std::vector<Tensor> cast_to_tensorlist(const std::vector<std::optional<Tensor>>& tensorlist) {
  std::vector<at::Tensor> tmpout;
  for(auto& opt_tensor: tensorlist) {
    if (opt_tensor.has_value()) {
      tmpout.push_back(opt_tensor.value());
    }
  }
  return tmpout;
}


class TORCH_API OptimizerParamState {
 public:
  OptimizerParamState() = default;
  OptimizerParamState(const OptimizerParamState&) = default;
  OptimizerParamState& operator=(const OptimizerParamState&) = default;
  OptimizerParamState(OptimizerParamState&&) noexcept = default;
  OptimizerParamState& operator=(OptimizerParamState&&) noexcept = default;
  virtual std::unique_ptr<OptimizerParamState> clone() const;
  virtual void serialize(torch::serialize::InputArchive& archive);
  virtual void serialize(torch::serialize::OutputArchive& archive) const;
  virtual ~OptimizerParamState() = default;
};

template <typename Derived>
class OptimizerCloneableParamState : public OptimizerParamState {
  std::unique_ptr<OptimizerParamState> clone() const override {
    return std::make_unique<Derived>(static_cast<const Derived&>(*this));
  }
};

class TORCH_API OptimizerOptions {
 public:
  OptimizerOptions() = default;
  OptimizerOptions(const OptimizerOptions&) = default;
  OptimizerOptions& operator=(const OptimizerOptions&) = default;
  OptimizerOptions(OptimizerOptions&&) noexcept = default;
  OptimizerOptions& operator=(OptimizerOptions&&) noexcept = default;
  virtual std::unique_ptr<OptimizerOptions> clone() const;
  virtual void serialize(torch::serialize::InputArchive& archive);
  virtual void serialize(torch::serialize::OutputArchive& archive) const;
  virtual ~OptimizerOptions() = default;
  virtual double get_lr() const;
  virtual void set_lr(const double lr);
};

template <typename Derived>
class OptimizerCloneableOptions : public OptimizerOptions {
 private:
  std::unique_ptr<OptimizerOptions> clone() const override {
    return std::make_unique<Derived>(static_cast<const Derived&>(*this));
  }
};

/// Stores parameters in the param_group and stores a pointer to the
/// OptimizerOptions
class TORCH_API OptimizerParamGroup {
 public:
  // NOTE: In order to store `OptimizerParamGroup` in a `std::vector`, it has to
  // be copy-constructible.
  OptimizerParamGroup(const OptimizerParamGroup& param_group)
      : params_(param_group.params()),
        options_(
            param_group.has_options() ? param_group.options().clone()
                                      : nullptr) {}
  OptimizerParamGroup(std::vector<Tensor> params)
      : params_(std::move(params)) {}
  OptimizerParamGroup(
      std::vector<Tensor> params,
      std::unique_ptr<OptimizerOptions> options)
      : params_(std::move(params)), options_(std::move(options)) {}

  OptimizerParamGroup& operator=(const OptimizerParamGroup& param_group) =
      delete;
  bool has_options() const;
  OptimizerOptions& options();
  const OptimizerOptions& options() const;
  void set_options(std::unique_ptr<OptimizerOptions> options);
  std::vector<Tensor>& params();
  const std::vector<Tensor>& params() const;

 protected:
  std::vector<Tensor> params_;
  std::unique_ptr<OptimizerOptions> options_;
};

class TORCH_API Optimizer {
 public:
  // The copy constructor is deleted, because the user should use the
  // `state_dict` / `load_state_dict` API to copy an optimizer instead.
  Optimizer(const Optimizer& optimizer) = delete;
  Optimizer(Optimizer&& optimizer) = default;

  explicit Optimizer(
      std::vector<OptimizerParamGroup> param_groups,
      std::unique_ptr<OptimizerOptions> defaults)
      : defaults_(std::move(defaults)) {
    for (const auto& param_group : param_groups) {
      add_param_group(param_group);
    }
  }

  /// Constructs the `Optimizer` from a vector of parameters.
  explicit Optimizer(
      std::vector<Tensor> parameters,
      std::unique_ptr<OptimizerOptions> defaults)
      : Optimizer(
            {OptimizerParamGroup(std::move(parameters))},
            std::move(defaults)){};

  /// Adds the given param_group to the optimizer's param_group list.
  void add_param_group(const OptimizerParamGroup& param_group);

  virtual ~Optimizer() = default;

  using LossClosure = std::function<Tensor()>;
  /// A loss function closure, which is expected to return the loss value.
  virtual Tensor step(LossClosure closure = nullptr) = 0;

  /// Adds the given vector of parameters to the optimizer's parameter list.
  void add_parameters(const std::vector<Tensor>& parameters);

  /// Zeros out the gradients of all parameters.
  void zero_grad(bool set_to_none = true);

  /// Provides a const reference to the parameters in the first param_group this
  /// optimizer holds.
  const std::vector<Tensor>& parameters() const noexcept;

  /// Provides a reference to the parameters in the first param_group this
  /// optimizer holds.
  std::vector<Tensor>& parameters() noexcept;

  /// Returns the number of parameters referenced by the optimizer.
  size_t size() const noexcept;

  OptimizerOptions& defaults() noexcept;

  const OptimizerOptions& defaults() const noexcept;

  /// Provides a reference to the param_groups this optimizer holds.
  std::vector<OptimizerParamGroup>& param_groups() noexcept;

  /// Provides a const reference to the param_groups this optimizer holds.
  const std::vector<OptimizerParamGroup>& param_groups() const noexcept;

  /// Provides a reference to the state this optimizer holds
  ska::flat_hash_map<void*, std::unique_ptr<OptimizerParamState>>&
  state() noexcept;

  /// Provides a const reference to the state this optimizer holds
  const ska::flat_hash_map<void*, std::unique_ptr<OptimizerParamState>>& state()
      const noexcept;

  /// Serializes the optimizer state into the given `archive`.
  virtual void save(serialize::OutputArchive& archive) const;

  /// Deserializes the optimizer state from the given `archive`.
  virtual void load(serialize::InputArchive& archive);

 protected:
  std::vector<OptimizerParamGroup> param_groups_;
  ska::flat_hash_map<void*, std::unique_ptr<OptimizerParamState>> state_;
  std::unique_ptr<OptimizerOptions> defaults_;
};

/* How do we decide whether to serialize undefined tensors or
  std::nullopt values into the output archive?
Answer: we strictly follow the behavior of Python API. To be more specific:

For optimizer options:
a) For undefined tensor: currently no tensor is used as an options argument in
Python API, so we don't need to worry about it now. b) For std::nullopt value:
we serialize std::nullopt values into the output archive, to follow the exact
same behavior as Python API.

For optimizer param state:
a) For undefined tensor: in param state, undefined tensor in C++ impl is
equivalent to missing key in Python impl. Since we don't serialize missing keys
in Python API, we skip undefined tensors when serializing the param state. b)
For std::nullopt value: in param state, std::nullopt value in C++ impl is
equivalent to missing key in Python impl. Since we don't serialize missing keys
in Python API, we skip std::nullopt values when serializing the param state. */

/// Serializes an `Optimizer` into an `OutputArchive`.
TORCH_API serialize::OutputArchive& operator<<(
    serialize::OutputArchive& archive,
    const Optimizer& optimizer);

/// Deserializes a `Tensor` from an `InputArchive`.
TORCH_API serialize::InputArchive& operator>>(
    serialize::InputArchive& archive,
    Optimizer& optimizer);

} // namespace optim
} // namespace torch
