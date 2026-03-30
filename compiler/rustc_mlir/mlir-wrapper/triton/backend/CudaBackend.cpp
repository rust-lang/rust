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

#include "mlir/IR/Builders.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Triple.h"

#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#include "nvidia/hopper/include/Transforms/Passes.h"
#include "nvidia/include/Dialect/NVGPU/IR/Dialect.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"

#include "CudaBackend.h"

#include <regex>

using namespace mlir;
using namespace triton;
using namespace nvidia_gpu;

CudaBackend::CudaBackend(std::string target, CudaOptions options)
    : Backend(target), m_options(options) {
  m_capability = Capability::Sm120; // AXM FIXME: Get capability from target
}

CudaBackend::~CudaBackend() {
  // nop
}

void CudaBackend::loadDialects(MLIRContext &context) {
  DialectRegistry registry;

  registry.insert<mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect,
                  mlir::triton::nvgpu::NVGPUDialect,
                  mlir::triton::nvws::NVWSDialect>();

  registerNVVMDialectTranslation(registry);

  context.appendDialectRegistry(registry);
}

Capability CudaBackend::getCapability() const { return m_capability; }

LogicalResult CudaBackend::makeLLVMIR(MLIRContext &context, ModuleOp module) {
  llvm::LLVMContext llvmContext;

  // Initialize LLVM targets (required for NVPTX/codegen)
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmParsers();
  llvm::InitializeAllAsmPrinters();

  // Address Sanitizer is only supported on the AMD backend; not applicable
  // when using the base Backend (no knobs). Subclasses can override and check
  // enable_asan and return failure() for NVIDIA.

  // Translate MLIR module (LLVM dialect) to LLVM IR module
  auto llvmMod =
      mlir::translateModuleToLLVMIR(module.getOperation(), llvmContext);
  if (!llvmMod) {
    llvm::errs() << "Failed to translate MLIR module to LLVM IR\n";
    return LogicalResult::failure();
  }

  // Set target triple for NVIDIA PTX
  auto triple = llvm::Triple("nvptx64-nvidia-cuda");
  llvmMod->setTargetTriple(triple);

  // Attach data layout for NVPTX64 (matches triple/capability; proc/features
  // would require TargetMachine if layout varied per SM).
  static const char nvptx64DataLayout[] =
      "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-"
      "f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:"
      "64";
  llvmMod->setDataLayout(llvm::DataLayout(nvptx64DataLayout));

  if (m_options.enable_reflect_ftz) {
    llvmMod->addModuleFlag(llvm::Module::Override, "nvvm-reflect-ftz", 1u);
  }

  // Link extern libs from options when the module has undefined symbols
  if (!m_options.extern_libs.empty()) {
    std::vector<std::string> libPaths;
    for (const auto &pair : m_options.extern_libs) {
      libPaths.push_back(pair.second);
    }

    auto result = linkExternLibs(llvmContext, *llvmMod, libPaths);
    if (failed(result)) {
      return result;
    }
  }

  // Do NOT run host O3 here: the pipeline has no TargetMachine for NVPTX, so
  // passes like the CGSCC devirt repeater mis-analyze GPU IR (and can loop on
  // infinite-loop / recursive functions that appear in Rust no_core stubs).
  // Optimization is done at the correct level by llvmTranslateToAsm via the
  // NVPTX TargetMachine's own CodeGen pipeline.

  // Serialize LLVM module to string and store in m_llvmir
  llvm::raw_string_ostream os(m_llvmir);
  llvmMod->print(os, nullptr);

  return LogicalResult::success();
}

LogicalResult CudaBackend::makeASM(MLIRContext &context, ModuleOp module) {
  // TODO: remove hardcoded values
  int ptx_version = 87;
  std::string triple = "nvptx64-nvidia-cuda";
  std::string proc = "sm_120a";
  std::string features = ""; // get_features(m_options, m_options.arch);

  std::vector<std::string> flags = {"nvptx-mad-wide-opt"};

  // 2. Translate LLVM module to assembly (PTX)
  // This part is pseudo-code, as actual translation will depend on LLVM API
  // presence.
  std::string src_asm = m_llvmir; // Assume m_llvmir contains the LLVM IR for
                                  // the module serialized right before
  std::string ret = llvmTranslateToAsm(src_asm, triple, proc, features, flags,
                                       m_options.enable_fp_fusion, false);
  if (ret.empty()) {
    llvm::errs() << "Failed to translate LLVM IR to PTX\n";
    llvm::errs() << "LLVM IR: " << src_asm << "\n";
    llvm::errs() << "Triple: " << triple << "\n";
    return LogicalResult::failure();
  }
  // 3. Find kernel name
  std::regex kernel_re(R"(\.visible \.entry ([a-zA-Z_][a-zA-Z0-9_]*))");
  std::smatch match;
  std::string kernel_name;
  if (std::regex_search(ret, match, kernel_re)) {
    kernel_name = match[1].str();
  } else {
    llvm::errs() << "Could not find kernel name in PTX output\n";
    return LogicalResult::failure();
  }

  // 4. Post-process version and target
  char ptx_major_minor[8];
  snprintf(ptx_major_minor, sizeof(ptx_major_minor), "%d.%d", ptx_version / 10,
           ptx_version % 10);
  ret = std::regex_replace(ret, std::regex(R"(\.version \d+\.\d+)"),
                           ".version " + std::string(ptx_major_minor));
  ret = std::regex_replace(ret, std::regex(R"(\.target sm_\d+)"),
                           ".target sm_" + std::to_string(m_capability));

  // 5. Remove debug flag if desired
  // No 'knobs' defined; always remove for now or leave as TODO.
  ret = std::regex_replace(ret, std::regex(R"(,\s*debug|debug,\s*)"), "");

  // 7. Save PTX (exposed via getASM())
  m_asm = std::move(ret);
  return LogicalResult::success();
}

std::optional<Error> CudaBackend::addCudaPass(PassManager &pm, CudaPass pass) {
  auto pass_fn = m_nvidia_pass_fns.find(pass);
  if (pass_fn == m_nvidia_pass_fns.end()) {
    m_last_error = std::make_optional(Error::InvalidPass);
    m_last_error_string = "Invalid nvidia pass";
    return m_last_error;
  }

  pm.addPass(pass_fn->second());
  return std::nullopt;
}

std::optional<Error> CudaBackend::addCudaPass(PassManager &pm, CudaPass pass,
                                              int arg0) {
  if (pass != CudaPass::ttnvgpuir_proxy_fence_insertion) {
    m_last_error = std::make_optional(Error::InvalidPass);
    m_last_error_string = "Invalid nvidia pass";
    return m_last_error;
  }

  pm.addPass(CudaBackend::createTritonGPUProxyFenceInsertionWrapper(arg0));
  return std::nullopt;
}

std::optional<Error> CudaBackend::addCudaPass(PassManager &pm, CudaPass pass,
                                              int arg0, int arg1) {
  if (pass != CudaPass::ttnvgpuir_to_llvmir) {
    m_last_error = std::make_optional(Error::InvalidPass);
    m_last_error_string = "Invalid nvidia pass";
    return m_last_error;
  }

  pm.addPass(createConvertTritonGPUToLLVMPass(arg0, arg1));
  return std::nullopt;
}

std::optional<Error> CudaBackend::addCudaPass(PassManager &pm, CudaPass pass,
                                              int arg0, bool arg1) {
  if (pass != CudaPass::hopper_warpspec) {
    m_last_error = std::make_optional(Error::InvalidPass);
    m_last_error_string = "Invalid nvidia pass";
    return m_last_error;
  }

  pm.addPass(createNVGPUWarpSpecialization({arg0, arg1}));
  return std::nullopt;
}

LogicalResult CudaBackend::makeTTIR(MLIRContext &context, ModuleOp module) {
  PassManager pm(&context);
  auto capability = getCapability();
  auto op = module.getOperation();

  addPass(pm, MlirPass::inliner);
  addPass(pm, MlirPass::ttir_rewrite_tensor_pointer);
  if (capability < 90) {
    addPass(pm, MlirPass::ttir_rewrite_tensor_descriptor_to_pointer);
  }
  addPass(pm, MlirPass::canonicalizer);
  addPass(pm, MlirPass::ttir_combine);
  addPass(pm, MlirPass::ttir_reorder_broadcast);
  addPass(pm, MlirPass::cse);
  addPass(pm, MlirPass::symbol_dce);
  addPass(pm, MlirPass::ttir_loop_unroll);

  return pm.run(op);
}

LogicalResult CudaBackend::makeTTGIR(MLIRContext &context, ModuleOp module) {
  PassManager pm(&context);
  auto capability = getCapability();
  auto capability_major = static_cast<int>(capability) / 10;
  auto op = module.getOperation();
  auto emuTF32 = (capability_major >= 8);

  if (m_options.maxnreg.has_value()) {
    auto maxnreg = m_options.maxnreg.value();
    OpBuilder builder(&context);

    op->setAttr("ttg.maxnreg", builder.getI32IntegerAttr(maxnreg));
  }

  std::string capability_str =
      std::string("cuda:").append(std::to_string(static_cast<int>(capability)));

  addPass(pm, MlirPass::ttir_convert_to_ttgpuir, capability_str,
          m_options.num_warps, 32, m_options.num_ctas);

  // optimize TTGIR
  addPass(pm, MlirPass::ttgpuir_coalesce);
  addPass(pm, MlirPass::ttgpuir_f32_dot_tc, emuTF32);

  addCudaPass(pm, CudaPass::ttnvgpuir_plan_cta);
  addPass(pm, MlirPass::ttgpuir_remove_layout_conversions);
  addPass(pm, MlirPass::ttgpuir_optimize_thread_locality);
  addPass(pm, MlirPass::ttgpuir_accelerate_matmul);
  addPass(pm, MlirPass::ttgpuir_remove_layout_conversions);
  addPass(pm, MlirPass::ttgpuir_optimize_dot_operands, capability_major >= 8);

  addCudaPass(pm, CudaPass::ttnvgpuir_optimize_descriptor_encoding);
  addPass(pm, MlirPass::ttir_loop_aware_cse);

  if (capability_major == 8 || capability_major == 9) {
    addPass(pm, MlirPass::ttgpuir_fuse_nested_loops);
    addPass(pm, MlirPass::canonicalizer);
    addPass(pm, MlirPass::ttir_triton_licm);
    addPass(pm, MlirPass::canonicalizer);
    addPass(pm, MlirPass::ttgpuir_combine_tensor_select_and_if);
    addCudaPass(pm, CudaPass::hopper_warpspec, m_options.num_stages,
                m_options.dump_enabled);
    addPass(pm, MlirPass::ttgpuir_assign_latencies, m_options.num_stages);
    addPass(pm, MlirPass::ttgpuir_schedule_loops);
    addPass(pm, MlirPass::ttgpuir_pipeline, m_options.num_stages,
            m_options.dump_enabled);
  } else if (capability_major >= 10) {
    addPass(pm, MlirPass::ttgpuir_fuse_nested_loops);
    addPass(pm, MlirPass::canonicalizer);
    addPass(pm, MlirPass::ttir_triton_licm);
    addPass(pm, MlirPass::ttgpuir_optimize_accumulator_init);
    addPass(pm, MlirPass::ttgpuir_hoist_tmem_alloc, false);

    addCudaPass(pm, CudaPass::ttnvgpuir_promote_lhs_to_tmem);
    addPass(pm, MlirPass::ttgpuir_assign_latencies, m_options.num_stages);
    addPass(pm, MlirPass::ttgpuir_schedule_loops);

    addPass(pm, MlirPass::ttgpuir_warp_specialize, m_options.num_stages);
    addPass(pm, MlirPass::ttgpuir_pipeline, m_options.num_stages,
            m_options.dump_enabled);
    addPass(pm, MlirPass::ttgpuir_optimize_partition_warps);
    addPass(pm, MlirPass::ttgpuir_combine_tensor_select_and_if);
    // hoist again and allow hoisting out of if statements
    addPass(pm, MlirPass::ttgpuir_hoist_tmem_alloc, true);
    addCudaPass(pm, CudaPass::ttnvgpuir_remove_tmem_tokens);
  } else {
    addPass(pm, MlirPass::ttir_triton_licm);
  }

  addPass(pm, MlirPass::canonicalizer);
  addPass(pm, MlirPass::ttir_loop_aware_cse);
  addPass(pm, MlirPass::ttgpuir_prefetch);
  addPass(pm, MlirPass::ttgpuir_optimize_dot_operands, capability_major >= 8);

  addPass(pm, MlirPass::ttgpuir_coalesce_async_copy);
  addCudaPass(pm, CudaPass::ttnvgpuir_optimize_tmem_layouts);
  if (capability_major >= 9) {
    addCudaPass(pm, CudaPass::ttnvgpuir_tma_lowering);
  }
  addPass(pm, MlirPass::ttgpuir_remove_layout_conversions);
  addCudaPass(pm, CudaPass::ttnvgpuir_interleave_tmem);

  addPass(pm, MlirPass::ttgpuir_reduce_data_duplication);
  addPass(pm, MlirPass::ttgpuir_reorder_instructions);
  addPass(pm, MlirPass::ttir_loop_aware_cse);
  addPass(pm, MlirPass::symbol_dce);

  addCudaPass(pm, CudaPass::ttnvgpuir_fence_insertion, capability);
  addCudaPass(pm, CudaPass::ttnvgpuir_lower_mma);

  addPass(pm, MlirPass::sccp);
  addPass(pm, MlirPass::cse);
  addPass(pm, MlirPass::canonicalizer);

  return pm.run(op);
}

LogicalResult CudaBackend::gluonToTTGIR(MLIRContext &context, ModuleOp module) {
  PassManager pm(&context);
  auto capability = getCapability();
  auto capability_major = static_cast<int>(capability) / 10;
  auto op = module.getOperation();

  addPass(pm, MlirPass::gluon_inliner);
  addPass(pm, MlirPass::gluon_infer_coalesced_encodings);
  addPass(pm, MlirPass::gluon_resolve_auto_encodings);
  addCudaPass(pm, CudaPass::ttnvgpuir_tma_lowering);
  addPass(pm, MlirPass::canonicalizer);
  addPass(pm, MlirPass::sccp);
  addPass(pm, MlirPass::ttir_loop_aware_cse);
  addPass(pm, MlirPass::gluon_canonicalizer);
  addPass(pm, MlirPass::ttgpuir_combine_tensor_select_and_if);

  return pm.run(op);
}

LogicalResult CudaBackend::makeLLIR(MLIRContext &context, ModuleOp module) {
  PassManager pm(&context);
  auto capability = getCapability();
  auto capability_major = static_cast<int>(capability) / 10;
  auto ptx_version = m_options.ptx_version.value_or(90);
  auto op = module.getOperation();

  addPass(pm, MlirPass::ttgpuir_combine_tensor_select_and_if);
  addPass(pm, MlirPass::ttgpuir_allocate_warp_groups);
  addPass(pm, MlirPass::scf_to_cf);
  addPass(pm, MlirPass::gluon_inliner);
  addPass(pm, MlirPass::ttgpuir_allocate_shared_memory_nv, capability,
          ptx_version);
  addCudaPass(pm, CudaPass::ttnvgpuir_allocate_tensor_memory);
  addCudaPass(pm, CudaPass::ttnvgpuir_check_matmul_two_cta);
  if (m_options.enable_experimental_consan) {
    addPass(pm, MlirPass::ttgpuir_concurrency_sanitizer);
  }

  addPass(pm, MlirPass::ttgpuir_allocate_global_scratch_memory);
  addCudaPass(pm, CudaPass::ttnvgpuir_proxy_fence_insertion, capability);

  if (m_options.instrumentation) {
    // AXM TODO: Implement instrumentation
    // CUDABackend.instrumentation.patch("ttgpuir_to_llvmir", pm, mod.context)
  }

  addCudaPass(pm, CudaPass::ttnvgpuir_to_llvmir, capability, ptx_version);
  addPass(pm, MlirPass::canonicalizer);
  addPass(pm, MlirPass::cse);
  addCudaPass(pm, CudaPass::ttnvgpuir_nvgpu_to_llvm);
  addCudaPass(pm, CudaPass::ttnvgpuir_warp_specialize_to_llvm);
  addPass(pm, MlirPass::canonicalizer);
  addPass(pm, MlirPass::cse);
  addPass(pm, MlirPass::symbol_dce);
  addPass(pm, MlirPass::nvvm_to_llvm);
  if (!m_options.disable_line_info) {
    addPass(pm, MlirPass::llvmir_di_scope);
  }

  if (m_options.instrumentation) {
    // AXM TODO: Implement instrumentation
    // CUDABackend.instrumentation.patch("llvmir_to_llvm", pm, mod.context)
  }

  return pm.run(op);

  //  # LLVM-IR (MLIR) -> LLVM-IR (LLVM)
  //       llvm.init_targets()
  //       context = llvm.context()
  //       if knobs.compilation.enable_asan:
  //           raise RuntimeError(
  //               "Address Sanitizer Error: Address sanitizer is currently only
  //               supported on the AMD backend")
  //       llvm_mod = llvm.to_module(mod, context)
  //       proc = sm_arch_from_capability(capability)
  //       features = get_features(options, self.target.arch)
  //       triple = 'nvptx64-nvidia-cuda'
  //       nvidia.set_short_ptr()
  //       llvm.attach_datalayout(llvm_mod, triple, proc, features)
  //       nvidia.set_nvvm_reflect_ftz(llvm_mod)

  //       if options.extern_libs and nvidia.has_extern_deps(llvm_mod):
  //           paths = [path for (name, path) in options.extern_libs]
  //           llvm.link_extern_libs(llvm_mod, paths)

  //       llvm.optimize_module(llvm_mod, llvm.OPTIMIZE_O3)

  //       # Get some metadata
  //       # warp-specialization mutates num_warps
  //       total_num_warps = src.get_int_attr("ttg.total-num-warps")
  //       if total_num_warps is not None:
  //           metadata["num_warps"] = total_num_warps
  //       metadata["shared"] = src.get_int_attr("ttg.shared")
  //       metadata["tmem_size"] = src.get_int_attr("ttg.tensor_memory_size")
  //       metadata["global_scratch_size"] =
  //       src.get_int_attr("ttg.global_scratch_memory_size")
  //       metadata["global_scratch_align"] =
  //       src.get_int_attr("ttg.global_scratch_memory_alignment")
  //       metadata["profile_scratch_size"] =
  //       src.get_int_attr("ttg.profile_scratch_memory_size") or 0
  //       metadata["profile_scratch_align"] =
  //       src.get_int_attr("ttg.profile_scratch_memory_alignment") or 1 ret =
  //       str(llvm_mod) del llvm_mod del context return ret
}

std::unique_ptr<mlir::Pass>
CudaBackend::createTritonGPUProxyFenceInsertionWrapper(int32_t capability) {
  ttng::TritonGPUProxyFenceInsertionOptions options;
  options.computeCapability = capability;
  return ttng::createTritonGPUProxyFenceInsertion(options);
}

LogicalResult CudaBackend::makeBIN(MLIRContext &context, ModuleOp module) {
  return LogicalResult::success();
}

LogicalResult
CudaBackend::linkExternLibs(llvm::LLVMContext &llvmContext,
                            llvm::Module &module,
                            const std::vector<std::string> &libPaths) {
  llvm::Linker linker(module);
  for (const auto &libPath : libPaths) {
    auto buf = llvm::MemoryBuffer::getFile(libPath);
    if (!buf) {
      llvm::errs() << "Failed to get memory buffer: " << libPath << "\n";
      return LogicalResult::failure();
    }

    auto src =
        llvm::getLazyBitcodeModule((*buf)->getMemBufferRef(), llvmContext);
    if (!src) {
      llvm::errs() << "Failed to get lazy bitcode module: " << libPath << "\n";
      return LogicalResult::failure();
    }

    if (linker.linkInModule(std::move(*src))) {
      llvm::errs() << "Failed to link extern library: " << libPath << "\n";
      return LogicalResult::failure();
    }
  }

  return LogicalResult::success();
}

/// Translates LLVM IR to NVPTX assembly (PTX) using the given triple, CPU,
/// and features. Returns the PTX string or empty on error.
std::string CudaBackend::llvmTranslateToAsm(
    const std::string &llvmIr, const std::string &tripleStr,
    const std::string &cpu, const std::string &features,
    const std::vector<std::string> & /*flags*/, bool /*enableFpFusion*/,
    bool /*verbose*/) {
  // Targets were already initialized in makeLLVMIR; no need to repeat.

  llvm::LLVMContext ctx;
  auto buf = llvm::MemoryBuffer::getMemBuffer(llvmIr, "<llvm-ir>");
  llvm::SMDiagnostic err;
  std::unique_ptr<llvm::Module> mod =
      llvm::parseIR(buf->getMemBufferRef(), err, ctx);
  if (!mod) {
    err.print("CudaBackend", llvm::errs());
    return {};
  }

  std::string targetError;
  llvm::Triple triple(llvm::Triple::normalize(tripleStr));
  const llvm::Target *target =
      llvm::TargetRegistry::lookupTarget(triple.getTriple(), targetError);
  if (!target) {
    llvm::errs() << targetError << "\n";
    return {};
  }

  llvm::TargetOptions opts;
  llvm::TargetMachine *tm = target->createTargetMachine(
      triple, cpu, features, opts, llvm::Reloc::Static, std::nullopt,
      llvm::CodeGenOptLevel::Default);
  if (!tm)
    return {};

  llvm::SmallVector<char, 0> asmBuf;
  {
    llvm::raw_svector_ostream os(asmBuf);
    llvm::legacy::PassManager pm;
    if (tm->addPassesToEmitFile(pm, os, nullptr,
                                llvm::CodeGenFileType::AssemblyFile)) {
      llvm::errs() << "Failed to add passes to emit file\n";
      delete tm;
      return {};
    }
    (void)pm.run(*mod);
  }
  delete tm;
  // Build return string after stream and pass manager are destroyed so
  // no shared state can cause use-after-free or hang when copying.
  return std::string(asmBuf.data(), asmBuf.size());
}
