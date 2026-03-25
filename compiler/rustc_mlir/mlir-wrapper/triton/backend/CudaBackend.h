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

#include <map>
#include <optional>
#include <string>
#include <tuple>
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

struct CudaOptions {
  int num_warps = 4;
  int num_ctas = 1;
  int num_stages = 3;
  std::optional<int> maxnreg = std::nullopt;
  std::tuple<int, int, int> cluster_dims = {1, 1, 1};
  std::optional<int> ptx_version = std::nullopt;
  std::optional<std::string> ptx_options = std::nullopt;
  // filename of a user-defined IR (*.{ttir|ttgir|llir|ptx})
  std::optional<std::string> ir_override = std::nullopt;
  bool enable_fp_fusion = true;
  bool launch_cooperative_grid = false;
  bool launch_pdl = false;
  std::vector<std::string> supported_fp8_dtypes = {"fp8e5", "fp8e4b15"};
  std::vector<std::string> deprecated_fp8_dot_operand_dtypes = {};
  std::string default_dot_input_precision = "tf32";
  std::vector<std::string> allowed_dot_input_precisions = {"tf32", "tf32x3",
                                                           "ieee"};
  std::optional<bool> max_num_imprecise_acc_default = std::nullopt;
  std::map<std::string, std::string> extern_libs = {};
  bool debug = false;
  std::string backend_name = "cuda";
  bool sanitize_overflow = true;
  std::optional<std::string> arch = std::nullopt;
  bool dump_enabled = false;
  bool enable_experimental_consan = false;
  bool instrumentation = false;
  bool disable_line_info = false;
  bool enable_reflect_ftz = false;
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
  CudaBackend(std::string target, CudaOptions options);

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

  CudaOptions m_options;
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
