/*
 * Copyright (c) 2026 Teenygrad.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef TRITON_CUDA_BACKEND_H
#define TRITON_CUDA_BACKEND_H

#include <stddef.h>
#include <stdint.h>
#include <string>
#include <vector>

#include "llvm/IR/Module.h"

#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h"
#include "nvidia/include/NVGPUToLLVM/Passes.h"
#include "nvidia/include/TritonNVIDIAGPUToLLVM/Passes.h"

#include "Backend.h"

namespace mlir {
namespace triton {

namespace ttng = mlir::triton::nvidia_gpu;

// ---------------------------------------------------------------------------
// FFI-safe helper types
// These map directly to #[repr(C)] Rust structs with equivalent fields.
// ---------------------------------------------------------------------------

/// An optional 32-bit signed integer.
struct OptionalI32 {
  bool has_value;
  int32_t value;
};

/// An optional boolean.
struct OptionalBool {
  bool has_value;
  bool value;
};

/// A 3-component integer dimension (x, y, z).
struct Dim3 {
  int32_t x;
  int32_t y;
  int32_t z;
};

// ---------------------------------------------------------------------------
// CUDA backend compile options (FFI-safe / repr(C))
//
// Uses only C-compatible types so it can be shared across the C/Rust FFI
// boundary (equivalent to a #[repr(C)] Rust struct).
//
// Ownership conventions for pointer fields:
//   - All `const char *` string fields are null-terminated C strings.
//     NULL means "use the backend default".
//   - All `const char **` array fields are paired with a `size_t` length.
//   - The *caller* is responsible for keeping all pointed-to data alive
//     for the duration of the compilation call.
// ---------------------------------------------------------------------------

/// FFI-safe compilation options for the CUDA (Triton NVIDIA) backend.
struct CudaCompileOptions {
  int32_t num_warps;
  int32_t num_ctas;
  int32_t num_stages;
  int32_t capability;
  OptionalI32 maxnreg;
  Dim3 cluster_dims;
  OptionalI32 ptx_version;
  const char *ptx_options; ///< NULL = not set
  const char *ir_override; ///< NULL = not set

  bool enable_fp_fusion;
  bool launch_cooperative_grid;
  bool launch_pdl;

  const char **supported_fp8_dtypes;
  size_t supported_fp8_dtypes_len;

  const char **deprecated_fp8_dot_operand_dtypes;
  size_t deprecated_fp8_dot_operand_dtypes_len;

  const char *default_dot_input_precision; ///< NULL = "tf32"
  const char **allowed_dot_input_precisions;
  size_t allowed_dot_input_precisions_len;

  OptionalBool max_num_imprecise_acc_default;

  /// Parallel key/value arrays describing external library (name, path) pairs.
  const char **extern_lib_keys;
  const char **extern_lib_values;
  size_t extern_libs_len;

  bool debug;
  const char *backend_name; ///< NULL = "cuda"
  bool sanitize_overflow;
  const char *arch; ///< NULL = not set
  bool dump_enabled;
  bool enable_experimental_consan;
  bool instrumentation;
  bool disable_line_info;
  bool enable_reflect_ftz;
};

enum Capability {
  Sm80 = 80,
  Sm86 = 86,
  Sm87 = 87,
  Sm89 = 89,
  Sm90 = 90,
  Sm100 = 100,
  Sm103 = 103,
  Sm110 = 110,
  Sm120 = 120,
};

enum CudaPass {
  // ttgpuir
  allocate_shared_memory_nv,

  // ttnvgpuir
  ttnvgpuir_to_llvmir,
  ttnvgpuir_plan_cta,
  ttnvgpuir_fence_insertion,
  ttnvgpuir_proxy_fence_insertion,
  ttnvgpuir_tma_lowering,
  ttnvgpuir_promote_lhs_to_tmem,
  ttnvgpuir_remove_tmem_tokens,
  ttnvgpuir_check_matmul_two_cta,
  ttnvgpuir_nvgpu_to_llvm,
  ttnvgpuir_warp_specialize_to_llvm,
  ttnvgpuir_allocate_tensor_memory,
  ttnvgpuir_lower_mma,
  ttnvgpuir_optimize_descriptor_encoding,
  ttnvgpuir_optimize_tmem_layouts,
  ttnvgpuir_interleave_tmem,

  // nvws
  nvws_lower_warp_group,
  nvws_lower_aref,
  nvws_assign_stage_phase,
  nvws_insert_tmem_aref,

  // hopper
  hopper_warpspec
};

class CudaBackend : public Backend {
public:
  CudaBackend(std::string target, CudaCompileOptions options);

  virtual ~CudaBackend();

  virtual void loadDialects(MLIRContext &context);

  virtual LogicalResult makeTTIR(MLIRContext &context,
                                 ModuleOp module) override;

  virtual LogicalResult makeTTGIR(MLIRContext &context,
                                  ModuleOp module) override;

  virtual LogicalResult gluonToTTGIR(MLIRContext &context,
                                     ModuleOp module) override;

  virtual LogicalResult makeLLIR(MLIRContext &context,
                                 ModuleOp module) override;

  virtual LogicalResult makeLLVMIR(MLIRContext &context,
                                   ModuleOp module) override;

  virtual LogicalResult makeASM(MLIRContext &context, ModuleOp module) override;

  virtual LogicalResult makeBIN(MLIRContext &context, ModuleOp module) override;

private:
  std::optional<Error> addCudaPass(PassManager &pm, CudaPass pass);

  std::optional<Error> addCudaPass(PassManager &pm, CudaPass pass, int arg0);

  std::optional<Error> addCudaPass(PassManager &pm, CudaPass pass, int arg0,
                                   int arg1);

  std::optional<Error> addCudaPass(PassManager &pm, CudaPass pass, int arg0,
                                   bool arg1);

  std::unique_ptr<mlir::Pass>
  createTritonGPUProxyFenceInsertionWrapper(int32_t capability);

  LogicalResult linkExternLibs(llvm::LLVMContext &llvmContext,
                               llvm::Module &module,
                               const std::vector<std::string> &libPaths);

  std::string llvmTranslateToAsm(const std::string &llvmIr,
                                 const std::string &tripleStr,
                                 const std::string &cpu,
                                 const std::string &features,
                                 const std::vector<std::string> & /*flags*/,
                                 bool /*enableFpFusion*/, bool /*verbose*/);

  CudaCompileOptions m_options;
  Capability m_capability;

  std::unordered_map<CudaPass, std::unique_ptr<Pass> (*)()> m_nvidia_pass_fns =
      {
          {ttnvgpuir_fence_insertion, ttng::createTritonGPUFenceInsertion},
          {ttnvgpuir_tma_lowering, ttng::createTritonNvidiaGPUTMALoweringPass},
          {ttnvgpuir_promote_lhs_to_tmem,
           ttng::createTritonNvidiaGPUPromoteLHSToTMemPass},
          {ttnvgpuir_remove_tmem_tokens,
           ttng::createTritonNvidiaGPURemoveTMEMTokensPass},
          {ttnvgpuir_check_matmul_two_cta,
           ttng::createTritonNvidiaGPUCheckMatmulTwoCTAPass},
          {ttnvgpuir_nvgpu_to_llvm, mlir::triton::createConvertNVGPUToLLVM},
          {ttnvgpuir_warp_specialize_to_llvm,
           mlir::triton::createConvertWarpSpecializeToLLVM},
          {ttnvgpuir_allocate_tensor_memory,
           ttng::createTritonTensorMemoryAllocationPass},
          {ttnvgpuir_lower_mma, ttng::createTritonNvidiaGPUMMALoweringPass},
          {ttnvgpuir_optimize_descriptor_encoding,
           ttng::createTritonNvidiaGPUOptimizeDescriptorEncodingPass},
          {ttnvgpuir_optimize_tmem_layouts,
           ttng::createTritonNvidiaGPUOptimizeTMemLayoutsPass},
          {ttnvgpuir_interleave_tmem,
           ttng::createTritonNvidiaGPUInterleaveTMemPass},
          {ttnvgpuir_plan_cta, ttng::createTritonNvidiaGPUPlanCTAPass},

          // nvws
          {nvws_lower_warp_group, mlir::triton::createNVWSLowerWarpGroup},
          {nvws_lower_aref, mlir::triton::createNVWSLowerAref},
          {nvws_assign_stage_phase, mlir::triton::createNVWSAssignStagePhase},
          {nvws_insert_tmem_aref, mlir::triton::createNVWSInsertTmemAref},
      };

  Capability getCapability() const;
};

} // namespace triton
} // namespace mlir

#endif /*! TRITON_CUDA_BACKEND_H */
