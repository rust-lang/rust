use std::ffi::CString;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::{fs, slice, str};

use libc::{c_char, c_int, c_uint, c_void, size_t};
use llvm::{
    IntPredicate,
    LLVMRustLLVMHasZlibCompressionForDebugSymbols, LLVMRustLLVMHasZstdCompressionForDebugSymbols,
};
use rustc_ast::expand::autodiff_attrs::{AutoDiffItem, DiffActivity, DiffMode};
use rustc_ast::expand::typetree::FncTree;
use rustc_codegen_ssa::back::link::ensure_removed;
use rustc_codegen_ssa::back::write::{
    BitcodeSection, CodegenContext, EmitObj, ModuleConfig, TargetMachineFactoryConfig,
    TargetMachineFactoryFn,
};
use rustc_codegen_ssa::traits::*;
use rustc_codegen_ssa::{CompiledModule, ModuleCodegen};
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::profiling::SelfProfilerRef;
use rustc_data_structures::small_c_str::SmallCStr;
use rustc_errors::{DiagCtxtHandle, FatalError, Level};
use rustc_fs_util::{link_or_copy, path_to_c_string};
use rustc_middle::ty::TyCtxt;
use rustc_session::config::{
    self, AutoDiff, Lto, OutputType, Passes, RemapPathScopeComponents, SplitDwarfKind,
    SwitchWithOptPath,
};
use rustc_session::Session;
use rustc_span::symbol::sym;
use rustc_span::InnerSpan;
use rustc_target::spec::{CodeModel, RelocModel, SanitizerSet, SplitDebuginfo, TlsModel};
use tracing::{debug, trace};

use crate::back::lto::ThinBuffer;
use crate::back::owned_target_machine::OwnedTargetMachine;
use crate::back::profiling::{
    selfprofile_after_pass_callback, selfprofile_before_pass_callback, LlvmSelfProfiler,
};
use crate::errors::{
    CopyBitcode, FromLlvmDiag, FromLlvmOptimizationDiag, LlvmError, UnknownCompression,
    WithLlvmError, WriteBytecode,
};
use crate::llvm::diagnostic::OptimizationDiagnosticKind;
use crate::llvm::{
    self, enzyme_rust_forward_diff, enzyme_rust_reverse_diff, AttributeKind,
    CreateEnzymeLogic, CreateTypeAnalysis, DiagnosticInfo, EnzymeLogicRef, EnzymeTypeAnalysisRef,
    FreeTypeAnalysis, LLVMAppendBasicBlockInContext, LLVMBuildCall2,
    LLVMBuildCondBr, LLVMBuildExtractValue, LLVMBuildICmp, LLVMBuildRet, LLVMBuildRetVoid,
    LLVMCountParams, LLVMCountStructElementTypes, LLVMCreateBuilderInContext,
    LLVMCreateStringAttribute, LLVMDisposeBuilder, LLVMDumpModule,
    LLVMGetFirstBasicBlock, LLVMGetFirstFunction,
    LLVMGetNextFunction, LLVMGetParams, LLVMGetReturnType,
    LLVMGetStringAttributeAtIndex, LLVMGlobalGetValueType, LLVMIsEnumAttribute,
    LLVMIsStringAttribute, LLVMMetadataAsValue, LLVMPositionBuilderAtEnd,
    LLVMRemoveStringAttributeAtIndex, LLVMRustAddEnumAttributeAtIndex,
    LLVMRustAddFunctionAttributes, LLVMRustDIGetInstMetadata,
    LLVMRustEraseInstBefore, LLVMRustEraseInstFromParent,
    LLVMRustGetEnumAttributeAtIndex, LLVMRustGetFunctionType, LLVMRustGetLastInstruction,
    LLVMRustGetTerminator, LLVMRustHasMetadata,
    LLVMRustRemoveEnumAttributeAtIndex,
    LLVMVerifyFunction,
    LLVMVoidTypeInContext, PassManager, Value,
};
use crate::type_::Type;
use crate::{base, common, llvm_util, DiffTypeTree, LlvmCodegenBackend, ModuleLlvm};

pub fn llvm_err<'a>(dcx: DiagCtxtHandle<'_>, err: LlvmError<'a>) -> FatalError {
    match llvm::last_error() {
        Some(llvm_err) => dcx.emit_almost_fatal(WithLlvmError(err, llvm_err)),
        None => dcx.emit_almost_fatal(err),
    }
}

pub fn write_output_file<'ll>(
    dcx: DiagCtxtHandle<'_>,
    target: &'ll llvm::TargetMachine,
    pm: &llvm::PassManager<'ll>,
    m: &'ll llvm::Module,
    output: &Path,
    dwo_output: Option<&Path>,
    file_type: llvm::FileType,
    self_profiler_ref: &SelfProfilerRef,
) -> Result<(), FatalError> {
    debug!("write_output_file output={:?} dwo_output={:?}", output, dwo_output);
    unsafe {
        let output_c = path_to_c_string(output);
        let dwo_output_c;
        let dwo_output_ptr = if let Some(dwo_output) = dwo_output {
            dwo_output_c = path_to_c_string(dwo_output);
            dwo_output_c.as_ptr()
        } else {
            std::ptr::null()
        };
        let result = llvm::LLVMRustWriteOutputFile(
            target,
            pm,
            m,
            output_c.as_ptr(),
            dwo_output_ptr,
            file_type,
        );

        // Record artifact sizes for self-profiling
        if result == llvm::LLVMRustResult::Success {
            let artifact_kind = match file_type {
                llvm::FileType::ObjectFile => "object_file",
                llvm::FileType::AssemblyFile => "assembly_file",
            };
            record_artifact_size(self_profiler_ref, artifact_kind, output);
            if let Some(dwo_file) = dwo_output {
                record_artifact_size(self_profiler_ref, "dwo_file", dwo_file);
            }
        }

        result.into_result().map_err(|()| llvm_err(dcx, LlvmError::WriteOutput { path: output }))
    }
}

pub fn create_informational_target_machine(
    sess: &Session,
    only_base_features: bool,
) -> OwnedTargetMachine {
    let config = TargetMachineFactoryConfig { split_dwarf_file: None, output_obj_file: None };
    // Can't use query system here quite yet because this function is invoked before the query
    // system/tcx is set up.
    let features = llvm_util::global_llvm_features(sess, false, only_base_features);
    target_machine_factory(sess, config::OptLevel::No, &features)(config)
        .unwrap_or_else(|err| llvm_err(sess.dcx(), err).raise())
}

pub fn create_target_machine(tcx: TyCtxt<'_>, mod_name: &str) -> OwnedTargetMachine {
    let split_dwarf_file = if tcx.sess.target_can_use_split_dwarf() {
        tcx.output_filenames(()).split_dwarf_path(
            tcx.sess.split_debuginfo(),
            tcx.sess.opts.unstable_opts.split_dwarf_kind,
            Some(mod_name),
        )
    } else {
        None
    };

    let output_obj_file =
        Some(tcx.output_filenames(()).temp_path(OutputType::Object, Some(mod_name)));
    let config = TargetMachineFactoryConfig { split_dwarf_file, output_obj_file };

    target_machine_factory(
        tcx.sess,
        tcx.backend_optimization_level(()),
        tcx.global_backend_features(()),
    )(config)
    .unwrap_or_else(|err| llvm_err(tcx.dcx(), err).raise())
}

pub fn to_llvm_opt_settings(
    cfg: config::OptLevel,
) -> (llvm::CodeGenOptLevel, llvm::CodeGenOptSize) {
    use self::config::OptLevel::*;
    match cfg {
        No => (llvm::CodeGenOptLevel::None, llvm::CodeGenOptSizeNone),
        Less => (llvm::CodeGenOptLevel::Less, llvm::CodeGenOptSizeNone),
        Default => (llvm::CodeGenOptLevel::Default, llvm::CodeGenOptSizeNone),
        Aggressive => (llvm::CodeGenOptLevel::Aggressive, llvm::CodeGenOptSizeNone),
        Size => (llvm::CodeGenOptLevel::Default, llvm::CodeGenOptSizeDefault),
        SizeMin => (llvm::CodeGenOptLevel::Default, llvm::CodeGenOptSizeAggressive),
    }
}

fn to_pass_builder_opt_level(cfg: config::OptLevel) -> llvm::PassBuilderOptLevel {
    use config::OptLevel::*;
    match cfg {
        No => llvm::PassBuilderOptLevel::O0,
        Less => llvm::PassBuilderOptLevel::O1,
        Default => llvm::PassBuilderOptLevel::O2,
        Aggressive => llvm::PassBuilderOptLevel::O3,
        Size => llvm::PassBuilderOptLevel::Os,
        SizeMin => llvm::PassBuilderOptLevel::Oz,
    }
}

fn to_llvm_relocation_model(relocation_model: RelocModel) -> llvm::RelocModel {
    match relocation_model {
        RelocModel::Static => llvm::RelocModel::Static,
        // LLVM doesn't have a PIE relocation model, it represents PIE as PIC with an extra attribute.
        RelocModel::Pic | RelocModel::Pie => llvm::RelocModel::PIC,
        RelocModel::DynamicNoPic => llvm::RelocModel::DynamicNoPic,
        RelocModel::Ropi => llvm::RelocModel::ROPI,
        RelocModel::Rwpi => llvm::RelocModel::RWPI,
        RelocModel::RopiRwpi => llvm::RelocModel::ROPI_RWPI,
    }
}

pub(crate) fn to_llvm_code_model(code_model: Option<CodeModel>) -> llvm::CodeModel {
    match code_model {
        Some(CodeModel::Tiny) => llvm::CodeModel::Tiny,
        Some(CodeModel::Small) => llvm::CodeModel::Small,
        Some(CodeModel::Kernel) => llvm::CodeModel::Kernel,
        Some(CodeModel::Medium) => llvm::CodeModel::Medium,
        Some(CodeModel::Large) => llvm::CodeModel::Large,
        None => llvm::CodeModel::None,
    }
}

pub fn target_machine_factory(
    sess: &Session,
    optlvl: config::OptLevel,
    target_features: &[String],
) -> TargetMachineFactoryFn<LlvmCodegenBackend> {
    let reloc_model = to_llvm_relocation_model(sess.relocation_model());

    let (opt_level, _) = to_llvm_opt_settings(optlvl);
    let use_softfp = sess.opts.cg.soft_float;

    let ffunction_sections =
        sess.opts.unstable_opts.function_sections.unwrap_or(sess.target.function_sections);
    let fdata_sections = ffunction_sections;
    let funique_section_names = !sess.opts.unstable_opts.no_unique_section_names;

    let code_model = to_llvm_code_model(sess.code_model());

    let mut singlethread = sess.target.singlethread;

    // On the wasm target once the `atomics` feature is enabled that means that
    // we're no longer single-threaded, or otherwise we don't want LLVM to
    // lower atomic operations to single-threaded operations.
    if singlethread && sess.target.is_like_wasm && sess.target_features.contains(&sym::atomics) {
        singlethread = false;
    }

    let triple = SmallCStr::new(&sess.target.llvm_target);
    let cpu = SmallCStr::new(llvm_util::target_cpu(sess));
    let features = CString::new(target_features.join(",")).unwrap();
    let abi = SmallCStr::new(&sess.target.llvm_abiname);
    let trap_unreachable =
        sess.opts.unstable_opts.trap_unreachable.unwrap_or(sess.target.trap_unreachable);
    let emit_stack_size_section = sess.opts.unstable_opts.emit_stack_sizes;

    let verbose_asm = sess.opts.unstable_opts.verbose_asm;
    let relax_elf_relocations =
        sess.opts.unstable_opts.relax_elf_relocations.unwrap_or(sess.target.relax_elf_relocations);

    let use_init_array =
        !sess.opts.unstable_opts.use_ctors_section.unwrap_or(sess.target.use_ctors_section);

    let path_mapping = sess.source_map().path_mapping().clone();

    let use_emulated_tls = matches!(sess.tls_model(), TlsModel::Emulated);

    // copy the exe path, followed by path all into one buffer
    // null terminating them so we can use them as null terminated strings
    let args_cstr_buff = {
        let mut args_cstr_buff: Vec<u8> = Vec::new();
        let exe_path = std::env::current_exe().unwrap_or_default();
        let exe_path_str = exe_path.into_os_string().into_string().unwrap_or_default();

        args_cstr_buff.extend_from_slice(exe_path_str.as_bytes());
        args_cstr_buff.push(0);

        for arg in sess.expanded_args.iter() {
            args_cstr_buff.extend_from_slice(arg.as_bytes());
            args_cstr_buff.push(0);
        }

        args_cstr_buff
    };

    let debuginfo_compression = sess.opts.debuginfo_compression.to_string();
    match sess.opts.debuginfo_compression {
        rustc_session::config::DebugInfoCompression::Zlib => {
            if !unsafe { LLVMRustLLVMHasZlibCompressionForDebugSymbols() } {
                sess.dcx().emit_warn(UnknownCompression { algorithm: "zlib" });
            }
        }
        rustc_session::config::DebugInfoCompression::Zstd => {
            if !unsafe { LLVMRustLLVMHasZstdCompressionForDebugSymbols() } {
                sess.dcx().emit_warn(UnknownCompression { algorithm: "zstd" });
            }
        }
        rustc_session::config::DebugInfoCompression::None => {}
    };
    let debuginfo_compression = SmallCStr::new(&debuginfo_compression);

    let file_name_display_preference =
        sess.filename_display_preference(RemapPathScopeComponents::DEBUGINFO);

    Arc::new(move |config: TargetMachineFactoryConfig| {
        let path_to_cstring_helper = |path: Option<PathBuf>| -> CString {
            let path = path.unwrap_or_default();
            let path = path_mapping
                .to_real_filename(path)
                .to_string_lossy(file_name_display_preference)
                .into_owned();
            CString::new(path).unwrap()
        };

        let split_dwarf_file = path_to_cstring_helper(config.split_dwarf_file);
        let output_obj_file = path_to_cstring_helper(config.output_obj_file);

        OwnedTargetMachine::new(
            &triple,
            &cpu,
            &features,
            &abi,
            code_model,
            reloc_model,
            opt_level,
            use_softfp,
            ffunction_sections,
            fdata_sections,
            funique_section_names,
            trap_unreachable,
            singlethread,
            verbose_asm,
            emit_stack_size_section,
            relax_elf_relocations,
            use_init_array,
            &split_dwarf_file,
            &output_obj_file,
            &debuginfo_compression,
            use_emulated_tls,
            &args_cstr_buff,
        )
    })
}

pub(crate) fn save_temp_bitcode(
    cgcx: &CodegenContext<LlvmCodegenBackend>,
    module: &ModuleCodegen<ModuleLlvm>,
    name: &str,
) {
    if !cgcx.save_temps {
        return;
    }
    unsafe {
        let ext = format!("{name}.bc");
        let cgu = Some(&module.name[..]);
        let path = cgcx.output_filenames.temp_path_ext(&ext, cgu);
        let cstr = path_to_c_string(&path);
        let llmod = module.module_llvm.llmod();
        llvm::LLVMWriteBitcodeToFile(llmod, cstr.as_ptr());
    }
}

/// In what context is a dignostic handler being attached to a codegen unit?
pub enum CodegenDiagnosticsStage {
    /// Prelink optimization stage.
    Opt,
    /// LTO/ThinLTO postlink optimization stage.
    LTO,
    /// Code generation.
    Codegen,
}

pub struct DiagnosticHandlers<'a> {
    data: *mut (&'a CodegenContext<LlvmCodegenBackend>, DiagCtxtHandle<'a>),
    llcx: &'a llvm::Context,
    old_handler: Option<&'a llvm::DiagnosticHandler>,
}

impl<'a> DiagnosticHandlers<'a> {
    pub fn new(
        cgcx: &'a CodegenContext<LlvmCodegenBackend>,
        dcx: DiagCtxtHandle<'a>,
        llcx: &'a llvm::Context,
        module: &ModuleCodegen<ModuleLlvm>,
        stage: CodegenDiagnosticsStage,
    ) -> Self {
        let remark_passes_all: bool;
        let remark_passes: Vec<CString>;
        match &cgcx.remark {
            Passes::All => {
                remark_passes_all = true;
                remark_passes = Vec::new();
            }
            Passes::Some(passes) => {
                remark_passes_all = false;
                remark_passes =
                    passes.iter().map(|name| CString::new(name.as_str()).unwrap()).collect();
            }
        };
        let remark_passes: Vec<*const c_char> =
            remark_passes.iter().map(|name: &CString| name.as_ptr()).collect();
        let remark_file = cgcx
            .remark_dir
            .as_ref()
            // Use the .opt.yaml file suffix, which is supported by LLVM's opt-viewer.
            .map(|dir| {
                let stage_suffix = match stage {
                    CodegenDiagnosticsStage::Codegen => "codegen",
                    CodegenDiagnosticsStage::Opt => "opt",
                    CodegenDiagnosticsStage::LTO => "lto",
                };
                dir.join(format!("{}.{stage_suffix}.opt.yaml", module.name))
            })
            .and_then(|dir| dir.to_str().and_then(|p| CString::new(p).ok()));

        let pgo_available = cgcx.opts.cg.profile_use.is_some();
        let data = Box::into_raw(Box::new((cgcx, dcx)));
        unsafe {
            let old_handler = llvm::LLVMRustContextGetDiagnosticHandler(llcx);
            llvm::LLVMRustContextConfigureDiagnosticHandler(
                llcx,
                diagnostic_handler,
                data.cast(),
                remark_passes_all,
                remark_passes.as_ptr(),
                remark_passes.len(),
                // The `as_ref()` is important here, otherwise the `CString` will be dropped
                // too soon!
                remark_file.as_ref().map(|dir| dir.as_ptr()).unwrap_or(std::ptr::null()),
                pgo_available,
            );
            DiagnosticHandlers { data, llcx, old_handler }
        }
    }
}

impl<'a> Drop for DiagnosticHandlers<'a> {
    fn drop(&mut self) {
        unsafe {
            llvm::LLVMRustContextSetDiagnosticHandler(self.llcx, self.old_handler);
            drop(Box::from_raw(self.data));
        }
    }
}

fn report_inline_asm(
    cgcx: &CodegenContext<LlvmCodegenBackend>,
    msg: String,
    level: llvm::DiagnosticLevel,
    mut cookie: u64,
    source: Option<(String, Vec<InnerSpan>)>,
) {
    // In LTO build we may get srcloc values from other crates which are invalid
    // since they use a different source map. To be safe we just suppress these
    // in LTO builds.
    if matches!(cgcx.lto, Lto::Fat | Lto::Thin) {
        cookie = 0;
    }
    let level = match level {
        llvm::DiagnosticLevel::Error => Level::Error,
        llvm::DiagnosticLevel::Warning => Level::Warning,
        llvm::DiagnosticLevel::Note | llvm::DiagnosticLevel::Remark => Level::Note,
    };
    cgcx.diag_emitter.inline_asm_error(cookie.try_into().unwrap(), msg, level, source);
}

unsafe extern "C" fn diagnostic_handler(info: &DiagnosticInfo, user: *mut c_void) {
    if user.is_null() {
        return;
    }
    let (cgcx, dcx) =
        unsafe { *(user as *const (&CodegenContext<LlvmCodegenBackend>, DiagCtxtHandle<'_>)) };

    match unsafe { llvm::diagnostic::Diagnostic::unpack(info) } {
        llvm::diagnostic::InlineAsm(inline) => {
            report_inline_asm(cgcx, inline.message, inline.level, inline.cookie, inline.source);
        }

        llvm::diagnostic::Optimization(opt) => {
            dcx.emit_note(FromLlvmOptimizationDiag {
                filename: &opt.filename,
                line: opt.line,
                column: opt.column,
                pass_name: &opt.pass_name,
                kind: match opt.kind {
                    OptimizationDiagnosticKind::OptimizationRemark => "success",
                    OptimizationDiagnosticKind::OptimizationMissed
                    | OptimizationDiagnosticKind::OptimizationFailure => "missed",
                    OptimizationDiagnosticKind::OptimizationAnalysis
                    | OptimizationDiagnosticKind::OptimizationAnalysisFPCommute
                    | OptimizationDiagnosticKind::OptimizationAnalysisAliasing => "analysis",
                    OptimizationDiagnosticKind::OptimizationRemarkOther => "other",
                },
                message: &opt.message,
            });
        }
        llvm::diagnostic::PGO(diagnostic_ref) | llvm::diagnostic::Linker(diagnostic_ref) => {
            let message = llvm::build_string(|s| unsafe {
                llvm::LLVMRustWriteDiagnosticInfoToString(diagnostic_ref, s)
            })
            .expect("non-UTF8 diagnostic");
            dcx.emit_warn(FromLlvmDiag { message });
        }
        llvm::diagnostic::Unsupported(diagnostic_ref) => {
            let message = llvm::build_string(|s| unsafe {
                llvm::LLVMRustWriteDiagnosticInfoToString(diagnostic_ref, s)
            })
            .expect("non-UTF8 diagnostic");
            dcx.emit_err(FromLlvmDiag { message });
        }
        llvm::diagnostic::UnknownDiagnostic(..) => {}
    }
}

fn get_pgo_gen_path(config: &ModuleConfig) -> Option<CString> {
    match config.pgo_gen {
        SwitchWithOptPath::Enabled(ref opt_dir_path) => {
            let path = if let Some(dir_path) = opt_dir_path {
                dir_path.join("default_%m.profraw")
            } else {
                PathBuf::from("default_%m.profraw")
            };

            Some(CString::new(format!("{}", path.display())).unwrap())
        }
        SwitchWithOptPath::Disabled => None,
    }
}

fn get_pgo_use_path(config: &ModuleConfig) -> Option<CString> {
    config
        .pgo_use
        .as_ref()
        .map(|path_buf| CString::new(path_buf.to_string_lossy().as_bytes()).unwrap())
}

fn get_pgo_sample_use_path(config: &ModuleConfig) -> Option<CString> {
    config
        .pgo_sample_use
        .as_ref()
        .map(|path_buf| CString::new(path_buf.to_string_lossy().as_bytes()).unwrap())
}

fn get_instr_profile_output_path(config: &ModuleConfig) -> Option<CString> {
    config.instrument_coverage.then(|| CString::new("default_%m_%p.profraw").unwrap())
}

pub(crate) unsafe fn llvm_optimize(
    cgcx: &CodegenContext<LlvmCodegenBackend>,
    dcx: DiagCtxtHandle<'_>,
    module: &ModuleCodegen<ModuleLlvm>,
    config: &ModuleConfig,
    opt_level: config::OptLevel,
    opt_stage: llvm::OptStage,
    first_run: bool,
    noop: bool,
) -> Result<(), FatalError> {
    if noop {
        return Ok(());
    }
    // Enzyme:
    // The whole point of compiler based AD is to differentiate optimized IR instead of unoptimized
    // source code. However, benchmarks show that optimizations increasing the code size
    // tend to reduce AD performance. Therefore deactivate them before AD, then differentiate the code
    // and finally re-optimize the module, now with all optimizations available.
    // TODO: In a future update we could figure out how to only optimize functions getting
    // differentiated.

    let unroll_loops;
    let vectorize_slp;
    let vectorize_loop;

    if first_run {
        unroll_loops = false;
        vectorize_slp = false;
        vectorize_loop = false;
    } else {
        unroll_loops =
            opt_level != config::OptLevel::Size && opt_level != config::OptLevel::SizeMin;
        vectorize_slp = config.vectorize_slp;
        vectorize_loop = config.vectorize_loop;
    }
    trace!(
        "Enzyme: Running with unroll_loops: {}, vectorize_slp: {}, vectorize_loop: {}",
        unroll_loops, vectorize_slp, vectorize_loop
    );
    let using_thin_buffers = opt_stage == llvm::OptStage::PreLinkThinLTO || config.bitcode_needed();
    let pgo_gen_path = get_pgo_gen_path(config);
    let pgo_use_path = get_pgo_use_path(config);
    let pgo_sample_use_path = get_pgo_sample_use_path(config);
    let is_lto = opt_stage == llvm::OptStage::ThinLTO || opt_stage == llvm::OptStage::FatLTO;
    let instr_profile_output_path = get_instr_profile_output_path(config);
    let sanitize_dataflow_abilist: Vec<_> = config
        .sanitizer_dataflow_abilist
        .iter()
        .map(|file| CString::new(file.as_str()).unwrap())
        .collect();
    let sanitize_dataflow_abilist_ptrs: Vec<_> =
        sanitize_dataflow_abilist.iter().map(|file| file.as_ptr()).collect();
    // Sanitizer instrumentation is only inserted during the pre-link optimization stage.
    let sanitizer_options = if !is_lto {
        Some(llvm::SanitizerOptions {
            sanitize_address: config.sanitizer.contains(SanitizerSet::ADDRESS),
            sanitize_address_recover: config.sanitizer_recover.contains(SanitizerSet::ADDRESS),
            sanitize_cfi: config.sanitizer.contains(SanitizerSet::CFI),
            sanitize_dataflow: config.sanitizer.contains(SanitizerSet::DATAFLOW),
            sanitize_dataflow_abilist: sanitize_dataflow_abilist_ptrs.as_ptr(),
            sanitize_dataflow_abilist_len: sanitize_dataflow_abilist_ptrs.len(),
            sanitize_kcfi: config.sanitizer.contains(SanitizerSet::KCFI),
            sanitize_memory: config.sanitizer.contains(SanitizerSet::MEMORY),
            sanitize_memory_recover: config.sanitizer_recover.contains(SanitizerSet::MEMORY),
            sanitize_memory_track_origins: config.sanitizer_memory_track_origins as c_int,
            sanitize_thread: config.sanitizer.contains(SanitizerSet::THREAD),
            sanitize_hwaddress: config.sanitizer.contains(SanitizerSet::HWADDRESS),
            sanitize_hwaddress_recover: config.sanitizer_recover.contains(SanitizerSet::HWADDRESS),
            sanitize_kernel_address: config.sanitizer.contains(SanitizerSet::KERNELADDRESS),
            sanitize_kernel_address_recover: config
                .sanitizer_recover
                .contains(SanitizerSet::KERNELADDRESS),
        })
    } else {
        None
    };

    let mut llvm_profiler = cgcx
        .prof
        .llvm_recording_enabled()
        .then(|| LlvmSelfProfiler::new(cgcx.prof.get_self_profiler().unwrap()));

    let llvm_selfprofiler =
        llvm_profiler.as_mut().map(|s| s as *mut _ as *mut c_void).unwrap_or(std::ptr::null_mut());

    let extra_passes = if !is_lto { config.passes.join(",") } else { "".to_string() };

    let llvm_plugins = config.llvm_plugins.join(",");

    let result = unsafe {
        llvm::LLVMRustOptimize(
            module.module_llvm.llmod(),
            &*module.module_llvm.tm,
            to_pass_builder_opt_level(opt_level),
            opt_stage,
            cgcx.opts.cg.linker_plugin_lto.enabled(),
            config.no_prepopulate_passes,
            config.verify_llvm_ir,
            using_thin_buffers,
            config.merge_functions,
            unroll_loops,
            vectorize_slp,
            vectorize_loop,
            config.no_builtins,
            config.emit_lifetime_markers,
            sanitizer_options.as_ref(),
            pgo_gen_path.as_ref().map_or(std::ptr::null(), |s| s.as_ptr()),
            pgo_use_path.as_ref().map_or(std::ptr::null(), |s| s.as_ptr()),
            config.instrument_coverage,
            instr_profile_output_path.as_ref().map_or(std::ptr::null(), |s| s.as_ptr()),
            config.instrument_gcov,
            pgo_sample_use_path.as_ref().map_or(std::ptr::null(), |s| s.as_ptr()),
            config.debug_info_for_profiling,
            llvm_selfprofiler,
            selfprofile_before_pass_callback,
            selfprofile_after_pass_callback,
            extra_passes.as_ptr().cast(),
            extra_passes.len(),
            llvm_plugins.as_ptr().cast(),
            llvm_plugins.len(),
        )
    };
    result.into_result().map_err(|()| llvm_err(dcx, LlvmError::RunLlvmPasses))
}

fn get_params(fnc: &Value) -> Vec<&Value> {
    unsafe {
        let param_num = LLVMCountParams(fnc) as usize;
        let mut fnc_args: Vec<&Value> = vec![];
        fnc_args.reserve(param_num);
        LLVMGetParams(fnc, fnc_args.as_mut_ptr());
        fnc_args.set_len(param_num);
        fnc_args
    }
}

// DESIGN:
// Today we have our placeholder function, and our Enzyme generated one.
// We create a wrapper function and delete the placeholder body. You can see the
// placeholder by running `cargo expand` on an autodiff invocation. We call the wrapper
// from the placeholder. This function is a bit longer, because it matches the Rust level
// autodiff macro with LLVM level Enzyme autodiff expectations.
//
// Think of computing the derivative with respect to &[f32] by marking it as duplicated.
// The user will then pass an extra &mut [f32] and we want add the derivative to that.
// On LLVM/Enzyme level, &[f32] however becomes `ptr, i64` and we mark ptr as duplicated,
// and i64 (len) as const. Enzyme will then expect `ptr, ptr, i64` as arguments. See how the
// second i64 from the mut slice isn't used? That's why we add a safety check to assert
// that the second (mut) slice is at least as long as the first (const) slice. Otherwise,
// Enzyme would write out of bounds if the first (const) slice is longer than the second.

unsafe fn create_call<'a>(
    tgt: &'a Value,
    src: &'a Value,
    llmod: &'a llvm::Module,
    llcx: &llvm::Context,
    // FIXME: Instead of recomputing the positions as we do it below, we should
    // start using this list of positions that indicate length integers.
    _size_positions: &[usize],
    ad: &[AutoDiff],
) {
    unsafe {
        // first, remove all calls from fnc
        let bb = LLVMGetFirstBasicBlock(tgt);
        let br = LLVMRustGetTerminator(bb);
        LLVMRustEraseInstFromParent(br);

        // now add a call to inner.
        // append call to src at end of bb.
        let f_ty = LLVMRustGetFunctionType(src);

        let inner_param_num = LLVMCountParams(src);
        let outer_param_num = LLVMCountParams(tgt);
        let outer_args: Vec<&Value> = get_params(tgt);
        let inner_args: Vec<&Value> = get_params(src);
        let mut call_args: Vec<&Value> = vec![];

        let mut safety_vals = vec![];
        let builder = LLVMCreateBuilderInContext(llcx);
        let last_inst = LLVMRustGetLastInstruction(bb).unwrap();
        LLVMPositionBuilderAtEnd(builder, bb);

        let safety_run_checks = !ad.contains(&AutoDiff::NoSafetyChecks);

        if inner_param_num == outer_param_num {
            call_args = outer_args;
        } else {
            trace!("Different number of args, adjusting");
            let mut outer_pos: usize = 0;
            let mut inner_pos: usize = 0;
            // copy over if they are identical.
            // If not, skip the outer arg (and assert it's int).
            while outer_pos < outer_param_num as usize {
                let inner_arg = inner_args[inner_pos];
                let outer_arg = outer_args[outer_pos];
                let inner_arg_ty = llvm::LLVMTypeOf(inner_arg);
                let outer_arg_ty = llvm::LLVMTypeOf(outer_arg);
                if inner_arg_ty == outer_arg_ty {
                    call_args.push(outer_arg);
                    inner_pos += 1;
                    outer_pos += 1;
                } else {
                    // out: rust: (&[f32], &mut [f32])
                    // out: llvm: (ptr, <>int1, ptr, int2)
                    // inner: (ptr, <>ptr, int)
                    // goal: call (ptr, ptr, int1), skipping int2
                    // we are here: <>
                    assert!(llvm::LLVMRustGetTypeKind(outer_arg_ty) == llvm::TypeKind::Integer);
                    assert!(llvm::LLVMRustGetTypeKind(inner_arg_ty) == llvm::TypeKind::Pointer);
                    let next_outer_arg = outer_args[outer_pos + 1];
                    let next_inner_arg = inner_args[inner_pos + 1];
                    let next_outer_arg_ty = llvm::LLVMTypeOf(next_outer_arg);
                    let next_inner_arg_ty = llvm::LLVMTypeOf(next_inner_arg);
                    assert!(
                        llvm::LLVMRustGetTypeKind(next_outer_arg_ty) == llvm::TypeKind::Pointer
                    );
                    assert!(
                        llvm::LLVMRustGetTypeKind(next_inner_arg_ty) == llvm::TypeKind::Integer
                    );
                    let next2_outer_arg = outer_args[outer_pos + 2];
                    let next2_outer_arg_ty = llvm::LLVMTypeOf(next2_outer_arg);
                    assert!(
                        llvm::LLVMRustGetTypeKind(next2_outer_arg_ty) == llvm::TypeKind::Integer
                    );
                    call_args.push(next_outer_arg);
                    call_args.push(outer_arg);

                    outer_pos += 3;
                    inner_pos += 2;

                    if safety_run_checks {
                        // Now we assert if int1 <= int2
                        let res = LLVMBuildICmp(
                            builder,
                            IntPredicate::IntULE as u32,
                            outer_arg,
                            next2_outer_arg,
                            "safety_check".as_ptr() as *const c_char,
                        );
                        safety_vals.push(res);
                    }
                }
            }
        }

        if inner_param_num as usize != call_args.len() {
            panic!(
                "Args len shouldn't differ. Please report this. {} : {}",
                inner_param_num,
                call_args.len()
            );
        }

        // Now add the safety checks.
        if !safety_vals.is_empty() {
            dbg!("Adding safety checks");
            assert!(safety_run_checks);
            // first we create one bb per check and two more for the fail and success case.
            let fail_bb = LLVMAppendBasicBlockInContext(
                llcx,
                tgt,
                "ad_safety_fail".as_ptr() as *const c_char,
            );
            let success_bb = LLVMAppendBasicBlockInContext(
                llcx,
                tgt,
                "ad_safety_success".as_ptr() as *const c_char,
            );
            for i in 1..safety_vals.len() {
                // 'or' all safety checks together
                // Doing some binary tree style or'ing here would be more efficient,
                // but I assume LLVM will opt it anyway
                let prev = safety_vals[i - 1];
                let curr = safety_vals[i];
                let res = llvm::LLVMBuildOr(
                    builder,
                    prev,
                    curr,
                    "safety_check".as_ptr() as *const c_char,
                );
                safety_vals[i] = res;
            }
            LLVMBuildCondBr(builder, safety_vals.last().unwrap(), success_bb, fail_bb);
            LLVMPositionBuilderAtEnd(builder, fail_bb);

            let panic_name: CString = get_panic_name(llmod);

            let mut arg_vec = vec![add_panic_msg_to_global(llmod, llcx)];

            let fnc1 = llvm::LLVMGetNamedFunction(llmod, panic_name.as_ptr() as *const c_char);
            assert!(fnc1.is_some());
            let fnc1 = fnc1.unwrap();
            let ty = LLVMRustGetFunctionType(fnc1);
            let call = LLVMBuildCall2(
                builder,
                ty,
                fnc1,
                arg_vec.as_mut_ptr(),
                arg_vec.len(),
                panic_name.as_ptr() as *const c_char,
            );
            llvm::LLVMSetTailCall(call, 1);
            llvm::LLVMBuildUnreachable(builder);
            LLVMPositionBuilderAtEnd(builder, success_bb);
        }

        let inner_fnc_name = llvm::get_value_name(src);
        let c_inner_fnc_name = CString::new(inner_fnc_name).unwrap();

        let mut struct_ret = LLVMBuildCall2(
            builder,
            f_ty,
            src,
            call_args.as_mut_ptr(),
            call_args.len(),
            c_inner_fnc_name.as_ptr(),
        );

        // Add dummy dbg info to our newly generated call, if we have any.
        let md_ty = llvm::LLVMGetMDKindIDInContext(
            llcx,
            "dbg".as_ptr() as *const c_char,
            "dbg".len() as c_uint,
        );


        if LLVMRustHasMetadata(last_inst, md_ty) {
            let md = LLVMRustDIGetInstMetadata(last_inst);
            let md_val = LLVMMetadataAsValue(llcx, md);
            let _md2 = llvm::LLVMSetMetadata(struct_ret, md_ty, md_val);
        } else {
            trace!("No dbg info");
        }

        // Now clean up placeholder code.
        LLVMRustEraseInstBefore(bb, last_inst);

        let f_return_type = LLVMGetReturnType(LLVMGlobalGetValueType(src));
        let f_is_struct = llvm::LLVMRustIsStructType(f_return_type);
        let void_type = LLVMVoidTypeInContext(llcx);
        // Now unwrap the struct_ret if it's actually a struct
        if f_is_struct {
            let num_elem_in_ret_struct = LLVMCountStructElementTypes(f_return_type);
            if num_elem_in_ret_struct == 1 {
                let inner_grad_name = "foo".to_string();
                let c_inner_grad_name = CString::new(inner_grad_name).unwrap();
                struct_ret =
                    LLVMBuildExtractValue(builder, struct_ret, 0, c_inner_grad_name.as_ptr());
            }
        }
        if f_return_type != void_type {
            let _ret = LLVMBuildRet(builder, struct_ret);
        } else {
            let _ret = LLVMBuildRetVoid(builder);
        }
        LLVMDisposeBuilder(builder);
        let _fnc_ok =
            LLVMVerifyFunction(tgt, llvm::LLVMVerifierFailureAction::LLVMAbortProcessAction);
    }
}

unsafe fn get_panic_name(llmod: &llvm::Module) -> CString {
    // The names are mangled and their ending changes based on a hash, so just take whichever.
    let mut f = unsafe { LLVMGetFirstFunction(llmod) };
    loop {
        if let Some(lf) = f {
            f = unsafe { LLVMGetNextFunction(lf) };
            let fnc_name = llvm::get_value_name(lf);
            let fnc_name: String = String::from_utf8(fnc_name.to_vec()).unwrap();
            if fnc_name.starts_with("_ZN4core9panicking14panic_explicit") {
                return CString::new(fnc_name).unwrap();
            } else if fnc_name.starts_with("_RN4core9panicking14panic_explicit") {
                return CString::new(fnc_name).unwrap();
            }
        } else {
            break;
        }
    }
    panic!("Could not find panic function");
}

// This code is called when Enzyme detects at runtime that one of the safety invariants is violated.
// For now we only check if shadow arguments are large enough. In this case we look for Rust panic
// functions in the module and call it. Due to hashing we can't hardcode the panic function name.
// Note: This worked even for panic=abort tests so seems solid enough for now.
// FIXME: Pick a panic function which allows displaying an error message.
// FIXME: We probably want to keep a handle at higher level and pass it down instead of searching.
unsafe fn add_panic_msg_to_global<'a>(
    llmod: &'a llvm::Module,
    llcx: &'a llvm::Context,
) -> &'a llvm::Value {
    unsafe {
        use llvm::*;

        // Convert the message to a CString
        let msg = "autodiff safety check failed!";
        let cmsg = CString::new(msg).unwrap();

        let msg_global_name = "ad_safety_msg".to_string();
        let cmsg_global_name = CString::new(msg_global_name).unwrap();

        // Get the length of the message
        let msg_len = msg.len();

        // Create the array type
        let i8_array_type = LLVMArrayType2(LLVMInt8TypeInContext(llcx), msg_len as u64);

        // Create the string constant
        let _string_const_val =
            LLVMConstStringInContext2(llcx, cmsg.as_ptr() as *const i8, msg_len as usize, 0);

        // Create the array initializer
        let mut array_elems: Vec<_> = Vec::with_capacity(msg_len);
        for i in 0..msg_len {
            let char_value =
                LLVMConstInt(LLVMInt8TypeInContext(llcx), cmsg.as_bytes()[i] as u64, 0);
            array_elems.push(char_value);
        }
        let array_initializer =
            LLVMConstArray2(LLVMInt8TypeInContext(llcx), array_elems.as_mut_ptr(), msg_len as u64);

        // Create the struct type
        let global_type = LLVMStructTypeInContext(llcx, [i8_array_type].as_mut_ptr(), 1, 0);

        // Create the struct initializer
        let struct_initializer =
            LLVMConstStructInContext(llcx, [array_initializer].as_mut_ptr(), 1, 0);

        // Add the global variable to the module
        let global_var = LLVMAddGlobal(llmod, global_type, cmsg_global_name.as_ptr() as *const i8);
        LLVMRustSetLinkage(global_var, Linkage::PrivateLinkage);
        LLVMSetInitializer(global_var, struct_initializer);

        global_var
    }
}
use rustc_errors::DiagCtxt;

// As unsafe as it can be.
#[allow(unused_variables)]
#[allow(unused)]
pub(crate) unsafe fn enzyme_ad(
    llmod: &llvm::Module,
    llcx: &llvm::Context,
    diag_handler: &DiagCtxt,
    item: AutoDiffItem,
    logic_ref: EnzymeLogicRef,
    ad: &[AutoDiff],
) -> Result<(), FatalError> {
    let autodiff_mode = item.attrs.mode;
    let rust_name = item.source;
    let rust_name2 = &item.target;

    let args_activity = item.attrs.input_activity.clone();
    let ret_activity: DiffActivity = item.attrs.ret_activity;

    // get target and source function
    let name = CString::new(rust_name.to_owned()).unwrap();
    let name2 = CString::new(rust_name2.clone()).unwrap();
    let src_fnc_opt = unsafe { llvm::LLVMGetNamedFunction(llmod, name.as_c_str().as_ptr()) };
    let src_fnc = match src_fnc_opt {
        Some(x) => x,
        None => {
            return Err(llvm_err(
                diag_handler.handle(),
                LlvmError::PrepareAutoDiff {
                    src: rust_name.to_owned(),
                    target: rust_name2.to_owned(),
                    error: "could not find src function".to_owned(),
                },
            ));
        }
    };
    let target_fnc_opt = unsafe { llvm::LLVMGetNamedFunction(llmod, name2.as_ptr()) };
    let target_fnc = match target_fnc_opt {
        Some(x) => x,
        None => {
            return Err(llvm_err(
                diag_handler.handle(),
                LlvmError::PrepareAutoDiff {
                    src: rust_name.to_owned(),
                    target: rust_name2.to_owned(),
                    error: "could not find target function".to_owned(),
                },
            ));
        }
    };
    let src_num_args = unsafe { llvm::LLVMCountParams(src_fnc) };
    let target_num_args = unsafe { llvm::LLVMCountParams(target_fnc) };
    // A really simple check
    assert!(src_num_args <= target_num_args);

    let type_analysis: EnzymeTypeAnalysisRef =
        unsafe { CreateTypeAnalysis(logic_ref, std::ptr::null_mut(), std::ptr::null_mut(), 0) };

    llvm::set_strict_aliasing(false);

    if ad.contains(&AutoDiff::PrintTA) {
        llvm::set_print_type(true);
    }
    if ad.contains(&AutoDiff::PrintTA) {
        llvm::set_print_type(true);
    }
    if ad.contains(&AutoDiff::PrintPerf) {
        llvm::set_print_perf(true);
    }
    if ad.contains(&AutoDiff::Print) {
        llvm::set_print(true);
    }

    let mode = match autodiff_mode {
        DiffMode::Forward => DiffMode::Forward,
        DiffMode::Reverse => DiffMode::Reverse,
        DiffMode::ForwardFirst => DiffMode::Forward,
        DiffMode::ReverseFirst => DiffMode::Reverse,
        _ => unreachable!(),
    };

    unsafe {
        let void_type = LLVMVoidTypeInContext(llcx);
        let return_type = LLVMGetReturnType(LLVMGlobalGetValueType(src_fnc));
        let void_ret = void_type == return_type;
        let mut tmp = match mode {
            DiffMode::Forward => enzyme_rust_forward_diff(
                logic_ref,
                type_analysis,
                src_fnc,
                args_activity,
                ret_activity,
                void_ret,
            ),
            DiffMode::Reverse => enzyme_rust_reverse_diff(
                logic_ref,
                type_analysis,
                src_fnc,
                args_activity,
                ret_activity,
            ),
            _ => unreachable!(),
        };
        let mut res: &Value = tmp.0;
        let size_positions: Vec<usize> = tmp.1;

        let f_return_type = LLVMGetReturnType(LLVMGlobalGetValueType(res));

        create_call(target_fnc, res, llmod, llcx, &size_positions, ad);
        // TODO: implement drop for wrapper type?
        FreeTypeAnalysis(type_analysis);
    }

    Ok(())
}

#[allow(unused_unsafe)]
pub(crate) unsafe fn differentiate(
    module: &ModuleCodegen<ModuleLlvm>,
    cgcx: &CodegenContext<LlvmCodegenBackend>,
    diff_items: Vec<AutoDiffItem>,
    _typetrees: FxHashMap<String, DiffTypeTree>,
    config: &ModuleConfig,
) -> Result<(), FatalError> {
    for item in &diff_items {
        trace!("{}", item);
    }

    let llmod = module.module_llvm.llmod();
    let llcx = &module.module_llvm.llcx;
    let diag_handler = cgcx.create_dcx();

    llvm::set_strict_aliasing(false);

    let ad = &config.autodiff;

    if ad.contains(&AutoDiff::LooseTypes) {
        dbg!("Setting loose types to true");
        llvm::set_loose_types(true);
    }

    // Before dumping the module, we want all the tt to become part of the module.
    for (i, item) in diff_items.iter().enumerate() {
        let tt: FncTree = FncTree { args: item.inputs.clone(), ret: item.output.clone() };
        let name = CString::new(item.source.clone()).unwrap();
        let fn_def: &llvm::Value =
            unsafe { llvm::LLVMGetNamedFunction(llmod, name.as_ptr()).unwrap() };
        crate::builder::add_tt2(llmod, llcx, fn_def, tt);

        // Before dumping the module, we also might want to add dummy functions,  which will
        // trigger the LLVMEnzyme pass to run on them, if we invoke the opt binary.
        // This is super helpfull if we want to create a MWE bug reproducer, e.g. to run in
        // Enzyme's compiler explorer. TODO: Can we run llvm-extract on the module to remove all other functions?
        if ad.contains(&AutoDiff::OPT) {
            dbg!("Enable extra debug helper to debug Enzyme through the opt plugin");
            crate::builder::add_opt_dbg_helper(llmod, llcx, fn_def, item.attrs.clone(), i);
        }
    }

    if ad.contains(&AutoDiff::PrintModBefore) || ad.contains(&AutoDiff::OPT) {
        unsafe {
            LLVMDumpModule(llmod);
        }
    }

    if ad.contains(&AutoDiff::Inline) {
        dbg!("Setting inline to true");
        llvm::set_inline(true);
    }

    if ad.contains(&AutoDiff::RuntimeActivity) {
        dbg!("Setting runtime activity check to true");
        llvm::set_runtime_activity_check(true);
    }

    for val in ad {
        match &val {
            AutoDiff::TTDepth(depth) => {
                assert!(*depth >= 1);
                llvm::set_max_int_offset(*depth);
            }
            AutoDiff::TTWidth(width) => {
                assert!(*width >= 1);
                llvm::set_max_type_offset(*width);
            }
            _ => {}
        }
    }

    let differentiate = !diff_items.is_empty();
    let mut first_order_items: Vec<AutoDiffItem> = vec![];
    let mut higher_order_items: Vec<AutoDiffItem> = vec![];
    for item in diff_items {
        if item.attrs.mode == DiffMode::ForwardFirst || item.attrs.mode == DiffMode::ReverseFirst {
            first_order_items.push(item);
        } else {
            // default
            higher_order_items.push(item);
        }
    }

    let fnc_opt = ad.contains(&AutoDiff::EnableFncOpt);

    // If a function is a base for some higher order ad, always optimize
    let fnc_opt_base = true;
    let logic_ref_opt: EnzymeLogicRef = unsafe { CreateEnzymeLogic(fnc_opt_base as u8) };

    for item in first_order_items {
        let res =
            unsafe { enzyme_ad(llmod, llcx, &diag_handler.handle(), item, logic_ref_opt, ad) };
        assert!(res.is_ok());
    }

    // For the rest, follow the user choice on debug vs release.
    // Reuse the opt one if possible for better compile time (Enzyme internal caching).
    let logic_ref = match fnc_opt {
        true => {
            dbg!("Enable extra optimizations for Enzyme");
            logic_ref_opt
        }
        false => unsafe { CreateEnzymeLogic(fnc_opt as u8) },
    };
    for item in higher_order_items {
        let res = unsafe { enzyme_ad(llmod, llcx, &diag_handler.handle(), item, logic_ref, ad) };
        assert!(res.is_ok());
    }

    unsafe {
        let mut f = LLVMGetFirstFunction(llmod);
        loop {
            if let Some(lf) = f {
                f = LLVMGetNextFunction(lf);
                let myhwattr = "enzyme_hw";
                let attr = LLVMGetStringAttributeAtIndex(
                    lf,
                    c_uint::MAX,
                    myhwattr.as_ptr() as *const c_char,
                    myhwattr.as_bytes().len() as c_uint,
                );
                if LLVMIsStringAttribute(attr) {
                    LLVMRemoveStringAttributeAtIndex(
                        lf,
                        c_uint::MAX,
                        myhwattr.as_ptr() as *const c_char,
                        myhwattr.as_bytes().len() as c_uint,
                    );
                } else {
                    LLVMRustRemoveEnumAttributeAtIndex(
                        lf,
                        c_uint::MAX,
                        AttributeKind::SanitizeHWAddress,
                    );
                }
            } else {
                break;
            }
        }
        if ad.contains(&AutoDiff::PrintModAfterEnzyme) {
            LLVMDumpModule(llmod);
        }
    }

    if ad.contains(&AutoDiff::NoModOptAfter) || !differentiate {
        trace!("Skipping module optimization after automatic differentiation");
    } else {
        if let Some(opt_level) = config.opt_level {
            let opt_stage = match cgcx.lto {
                Lto::Fat => llvm::OptStage::PreLinkFatLTO,
                Lto::Thin | Lto::ThinLocal => llvm::OptStage::PreLinkThinLTO,
                _ if cgcx.opts.cg.linker_plugin_lto.enabled() => llvm::OptStage::PreLinkThinLTO,
                _ => llvm::OptStage::PreLinkNoLTO,
            };
            let mut first_run = false;
            dbg!("Running Module Optimization after differentiation");
            if ad.contains(&AutoDiff::NoVecUnroll) {
                // disables vectorization and loop unrolling
                first_run = true;
            }
            if ad.contains(&AutoDiff::AltPipeline) {
                dbg!("Running first postAD optimization");
                first_run = true;
            }
            let noop = false;
            unsafe {
                llvm_optimize(
                    cgcx,
                    diag_handler.handle(),
                    module,
                    config,
                    opt_level,
                    opt_stage,
                    first_run,
                    noop,
                )?
            };
        }
        if ad.contains(&AutoDiff::AltPipeline) {
            dbg!("Running Second postAD optimization");
            if let Some(opt_level) = config.opt_level {
                let opt_stage = match cgcx.lto {
                    Lto::Fat => llvm::OptStage::PreLinkFatLTO,
                    Lto::Thin | Lto::ThinLocal => llvm::OptStage::PreLinkThinLTO,
                    _ if cgcx.opts.cg.linker_plugin_lto.enabled() => llvm::OptStage::PreLinkThinLTO,
                    _ => llvm::OptStage::PreLinkNoLTO,
                };
                let mut first_run = false;
                dbg!("Running Module Optimization after differentiation");
                if ad.contains(&AutoDiff::NoVecUnroll) {
                    // enables vectorization and loop unrolling
                    first_run = false;
                }
                let noop = false;
                unsafe {
                    llvm_optimize(
                        cgcx,
                        diag_handler.handle(),
                        module,
                        config,
                        opt_level,
                        opt_stage,
                        first_run,
                        noop,
                    )?
                };
            }
        }
    }

    if ad.contains(&AutoDiff::PrintModAfterOpts) {
        unsafe {
            LLVMDumpModule(llmod);
        }
    }

    Ok(())
}

// Unsafe due to LLVM calls.
pub(crate) unsafe fn optimize(
    cgcx: &CodegenContext<LlvmCodegenBackend>,
    dcx: DiagCtxtHandle<'_>,
    module: &ModuleCodegen<ModuleLlvm>,
    config: &ModuleConfig,
) -> Result<(), FatalError> {
    let _timer = cgcx.prof.generic_activity_with_arg("LLVM_module_optimize", &*module.name);

    let llmod = module.module_llvm.llmod();
    let llcx = &*module.module_llvm.llcx;
    let _handlers = DiagnosticHandlers::new(cgcx, dcx, llcx, module, CodegenDiagnosticsStage::Opt);

    let module_name = module.name.clone();
    let module_name = Some(&module_name[..]);

    if config.emit_no_opt_bc {
        let out = cgcx.output_filenames.temp_path_ext("no-opt.bc", module_name);
        let out = path_to_c_string(&out);
        unsafe { llvm::LLVMWriteBitcodeToFile(llmod, out.as_ptr()) };
    }

    // This code enables Enzyme to differentiate code containing Rust enums.
    // By adding the SanitizeHWAddress attribute we prevent LLVM from Optimizing
    // away the enums and allows Enzyme to understand why a value can be of different types in
    // different code sections. We remove this attribute after Enzyme is done, to not affect the
    // rest of the compilation.
    // TODO: only enable this code when at least one function gets differentiated.
    unsafe {
        let mut f = LLVMGetFirstFunction(llmod);
        loop {
            if let Some(lf) = f {
                f = LLVMGetNextFunction(lf);
                let myhwattr = "enzyme_hw";
                let myhwv = "";
                let prevattr = LLVMRustGetEnumAttributeAtIndex(
                    lf,
                    c_uint::MAX,
                    AttributeKind::SanitizeHWAddress,
                );
                if LLVMIsEnumAttribute(prevattr) {
                    let attr = LLVMCreateStringAttribute(
                        llcx,
                        myhwattr.as_ptr() as *const c_char,
                        myhwattr.as_bytes().len() as c_uint,
                        myhwv.as_ptr() as *const c_char,
                        myhwv.as_bytes().len() as c_uint,
                    );
                    LLVMRustAddFunctionAttributes(lf, c_uint::MAX, &attr, 1);
                } else {
                    LLVMRustAddEnumAttributeAtIndex(
                        llcx,
                        lf,
                        c_uint::MAX,
                        AttributeKind::SanitizeHWAddress,
                    );
                }
            } else {
                break;
            }
        }
    }

    if let Some(opt_level) = config.opt_level {
        let opt_stage = match cgcx.lto {
            Lto::Fat => llvm::OptStage::PreLinkFatLTO,
            Lto::Thin | Lto::ThinLocal => llvm::OptStage::PreLinkThinLTO,
            _ if cgcx.opts.cg.linker_plugin_lto.enabled() => llvm::OptStage::PreLinkThinLTO,
            _ => llvm::OptStage::PreLinkNoLTO,
        };

        // Second run only relevant for AD
        let first_run = true;
        let noop = false;
        //if ad.contains(&AutoDiff::AltPipeline) {
        //    noop = true;
        //    dbg!("Skipping PreAD optimization");
        //} else {
        //    noop = false;
        //}
        return unsafe {
            llvm_optimize(cgcx, dcx, module, config, opt_level, opt_stage, first_run, noop)
        };
    }
    Ok(())
}

pub(crate) fn link(
    cgcx: &CodegenContext<LlvmCodegenBackend>,
    dcx: DiagCtxtHandle<'_>,
    mut modules: Vec<ModuleCodegen<ModuleLlvm>>,
) -> Result<ModuleCodegen<ModuleLlvm>, FatalError> {
    use super::lto::{Linker, ModuleBuffer};
    // Sort the modules by name to ensure deterministic behavior.
    modules.sort_by(|a, b| a.name.cmp(&b.name));
    let (first, elements) =
        modules.split_first().expect("Bug! modules must contain at least one module.");

    let mut linker = Linker::new(first.module_llvm.llmod());
    for module in elements {
        let _timer = cgcx.prof.generic_activity_with_arg("LLVM_link_module", &*module.name);
        let buffer = ModuleBuffer::new(module.module_llvm.llmod());
        linker
            .add(buffer.data())
            .map_err(|()| llvm_err(dcx, LlvmError::SerializeModule { name: &module.name }))?;
    }
    drop(linker);
    Ok(modules.remove(0))
}

pub(crate) unsafe fn codegen(
    cgcx: &CodegenContext<LlvmCodegenBackend>,
    dcx: DiagCtxtHandle<'_>,
    module: ModuleCodegen<ModuleLlvm>,
    config: &ModuleConfig,
) -> Result<CompiledModule, FatalError> {
    let _timer = cgcx.prof.generic_activity_with_arg("LLVM_module_codegen", &*module.name);
    {
        let llmod = module.module_llvm.llmod();
        let llcx = &*module.module_llvm.llcx;
        let tm = &*module.module_llvm.tm;
        let module_name = module.name.clone();
        let module_name = Some(&module_name[..]);
        let _handlers =
            DiagnosticHandlers::new(cgcx, dcx, llcx, &module, CodegenDiagnosticsStage::Codegen);

        if cgcx.msvc_imps_needed {
            create_msvc_imps(cgcx, llcx, llmod);
        }

        // A codegen-specific pass manager is used to generate object
        // files for an LLVM module.
        //
        // Apparently each of these pass managers is a one-shot kind of
        // thing, so we create a new one for each type of output. The
        // pass manager passed to the closure should be ensured to not
        // escape the closure itself, and the manager should only be
        // used once.
        unsafe fn with_codegen<'ll, F, R>(
            tm: &'ll llvm::TargetMachine,
            llmod: &'ll llvm::Module,
            no_builtins: bool,
            f: F,
        ) -> R
        where
            F: FnOnce(&'ll mut PassManager<'ll>) -> R,
        {
            unsafe {
                let cpm = llvm::LLVMCreatePassManager();
                llvm::LLVMAddAnalysisPasses(tm, cpm);
                llvm::LLVMRustAddLibraryInfo(cpm, llmod, no_builtins);
                f(cpm)
            }
        }

        // Two things to note:
        // - If object files are just LLVM bitcode we write bitcode, copy it to
        //   the .o file, and delete the bitcode if it wasn't otherwise
        //   requested.
        // - If we don't have the integrated assembler then we need to emit
        //   asm from LLVM and use `gcc` to create the object file.

        let bc_out = cgcx.output_filenames.temp_path(OutputType::Bitcode, module_name);
        let bc_summary_out =
            cgcx.output_filenames.temp_path(OutputType::ThinLinkBitcode, module_name);
        let obj_out = cgcx.output_filenames.temp_path(OutputType::Object, module_name);

        if config.bitcode_needed() {
            let _timer = cgcx
                .prof
                .generic_activity_with_arg("LLVM_module_codegen_make_bitcode", &*module.name);
            let thin = ThinBuffer::new(llmod, config.emit_thin_lto, config.emit_thin_lto_summary);
            let data = thin.data();

            if let Some(bitcode_filename) = bc_out.file_name() {
                cgcx.prof.artifact_size(
                    "llvm_bitcode",
                    bitcode_filename.to_string_lossy(),
                    data.len() as u64,
                );
            }

            if config.emit_thin_lto_summary
                && let Some(thin_link_bitcode_filename) = bc_summary_out.file_name()
            {
                let summary_data = thin.thin_link_data();
                cgcx.prof.artifact_size(
                    "llvm_bitcode_summary",
                    thin_link_bitcode_filename.to_string_lossy(),
                    summary_data.len() as u64,
                );

                let _timer = cgcx.prof.generic_activity_with_arg(
                    "LLVM_module_codegen_emit_bitcode_summary",
                    &*module.name,
                );
                if let Err(err) = fs::write(&bc_summary_out, summary_data) {
                    dcx.emit_err(WriteBytecode { path: &bc_summary_out, err });
                }
            }

            if config.emit_bc || config.emit_obj == EmitObj::Bitcode {
                let _timer = cgcx
                    .prof
                    .generic_activity_with_arg("LLVM_module_codegen_emit_bitcode", &*module.name);
                if let Err(err) = fs::write(&bc_out, data) {
                    dcx.emit_err(WriteBytecode { path: &bc_out, err });
                }
            }

            if config.emit_obj == EmitObj::ObjectCode(BitcodeSection::Full) {
                let _timer = cgcx
                    .prof
                    .generic_activity_with_arg("LLVM_module_codegen_embed_bitcode", &*module.name);
                unsafe {
                    embed_bitcode(cgcx, llcx, llmod, &config.bc_cmdline, data);
                }
            }
        }

        if config.emit_ir {
            let _timer =
                cgcx.prof.generic_activity_with_arg("LLVM_module_codegen_emit_ir", &*module.name);
            let out = cgcx.output_filenames.temp_path(OutputType::LlvmAssembly, module_name);
            let out_c = path_to_c_string(&out);

            extern "C" fn demangle_callback(
                input_ptr: *const c_char,
                input_len: size_t,
                output_ptr: *mut c_char,
                output_len: size_t,
            ) -> size_t {
                let input =
                    unsafe { slice::from_raw_parts(input_ptr as *const u8, input_len as usize) };

                let Ok(input) = str::from_utf8(input) else { return 0 };

                let output = unsafe {
                    slice::from_raw_parts_mut(output_ptr as *mut u8, output_len as usize)
                };
                let mut cursor = io::Cursor::new(output);

                let Ok(demangled) = rustc_demangle::try_demangle(input) else { return 0 };

                if write!(cursor, "{demangled:#}").is_err() {
                    // Possible only if provided buffer is not big enough
                    return 0;
                }

                cursor.position() as size_t
            }

            let result =
                unsafe { llvm::LLVMRustPrintModule(llmod, out_c.as_ptr(), demangle_callback) };

            if result == llvm::LLVMRustResult::Success {
                record_artifact_size(&cgcx.prof, "llvm_ir", &out);
            }

            result.into_result().map_err(|()| llvm_err(dcx, LlvmError::WriteIr { path: &out }))?;
        }

        if config.emit_asm {
            let _timer =
                cgcx.prof.generic_activity_with_arg("LLVM_module_codegen_emit_asm", &*module.name);
            let path = cgcx.output_filenames.temp_path(OutputType::Assembly, module_name);

            // We can't use the same module for asm and object code output,
            // because that triggers various errors like invalid IR or broken
            // binaries. So we must clone the module to produce the asm output
            // if we are also producing object code.
            let llmod = if let EmitObj::ObjectCode(_) = config.emit_obj {
                unsafe { llvm::LLVMCloneModule(llmod) }
            } else {
                llmod
            };
            unsafe {
                with_codegen(tm, llmod, config.no_builtins, |cpm| {
                    write_output_file(
                        dcx,
                        tm,
                        cpm,
                        llmod,
                        &path,
                        None,
                        llvm::FileType::AssemblyFile,
                        &cgcx.prof,
                    )
                })?;
            }
        }

        match config.emit_obj {
            EmitObj::ObjectCode(_) => {
                let _timer = cgcx
                    .prof
                    .generic_activity_with_arg("LLVM_module_codegen_emit_obj", &*module.name);

                let dwo_out = cgcx.output_filenames.temp_path_dwo(module_name);
                let dwo_out = match (cgcx.split_debuginfo, cgcx.split_dwarf_kind) {
                    // Don't change how DWARF is emitted when disabled.
                    (SplitDebuginfo::Off, _) => None,
                    // Don't provide a DWARF object path if split debuginfo is enabled but this is
                    // a platform that doesn't support Split DWARF.
                    _ if !cgcx.target_can_use_split_dwarf => None,
                    // Don't provide a DWARF object path in single mode, sections will be written
                    // into the object as normal but ignored by linker.
                    (_, SplitDwarfKind::Single) => None,
                    // Emit (a subset of the) DWARF into a separate dwarf object file in split
                    // mode.
                    (_, SplitDwarfKind::Split) => Some(dwo_out.as_path()),
                };

                unsafe {
                    with_codegen(tm, llmod, config.no_builtins, |cpm| {
                        write_output_file(
                            dcx,
                            tm,
                            cpm,
                            llmod,
                            &obj_out,
                            dwo_out,
                            llvm::FileType::ObjectFile,
                            &cgcx.prof,
                        )
                    })?;
                }
            }

            EmitObj::Bitcode => {
                debug!("copying bitcode {:?} to obj {:?}", bc_out, obj_out);
                if let Err(err) = link_or_copy(&bc_out, &obj_out) {
                    dcx.emit_err(CopyBitcode { err });
                }

                if !config.emit_bc {
                    debug!("removing_bitcode {:?}", bc_out);
                    ensure_removed(dcx, &bc_out);
                }
            }

            EmitObj::None => {}
        }

        record_llvm_cgu_instructions_stats(&cgcx.prof, llmod);
    }

    // `.dwo` files are only emitted if:
    //
    // - Object files are being emitted (i.e. bitcode only or metadata only compilations will not
    //   produce dwarf objects, even if otherwise enabled)
    // - Target supports Split DWARF
    // - Split debuginfo is enabled
    // - Split DWARF kind is `split` (i.e. debuginfo is split into `.dwo` files, not different
    //   sections in the `.o` files).
    let dwarf_object_emitted = matches!(config.emit_obj, EmitObj::ObjectCode(_))
        && cgcx.target_can_use_split_dwarf
        && cgcx.split_debuginfo != SplitDebuginfo::Off
        && cgcx.split_dwarf_kind == SplitDwarfKind::Split;
    Ok(module.into_compiled_module(
        config.emit_obj != EmitObj::None,
        dwarf_object_emitted,
        config.emit_bc,
        config.emit_asm,
        config.emit_ir,
        &cgcx.output_filenames,
    ))
}

fn create_section_with_flags_asm(section_name: &str, section_flags: &str, data: &[u8]) -> Vec<u8> {
    let mut asm = format!(".section {section_name},\"{section_flags}\"\n").into_bytes();
    asm.extend_from_slice(b".ascii \"");
    asm.reserve(data.len());
    for &byte in data {
        if byte == b'\\' || byte == b'"' {
            asm.push(b'\\');
            asm.push(byte);
        } else if byte < 0x20 || byte >= 0x80 {
            // Avoid non UTF-8 inline assembly. Use octal escape sequence, because it is fixed
            // width, while hex escapes will consume following characters.
            asm.push(b'\\');
            asm.push(b'0' + ((byte >> 6) & 0x7));
            asm.push(b'0' + ((byte >> 3) & 0x7));
            asm.push(b'0' + ((byte >> 0) & 0x7));
        } else {
            asm.push(byte);
        }
    }
    asm.extend_from_slice(b"\"\n");
    asm
}

fn target_is_apple(cgcx: &CodegenContext<LlvmCodegenBackend>) -> bool {
    cgcx.opts.target_triple.triple().contains("-ios")
        || cgcx.opts.target_triple.triple().contains("-darwin")
        || cgcx.opts.target_triple.triple().contains("-tvos")
        || cgcx.opts.target_triple.triple().contains("-watchos")
        || cgcx.opts.target_triple.triple().contains("-visionos")
}

fn target_is_aix(cgcx: &CodegenContext<LlvmCodegenBackend>) -> bool {
    cgcx.opts.target_triple.triple().contains("-aix")
}

//FIXME use c string literals here too
pub(crate) fn bitcode_section_name(cgcx: &CodegenContext<LlvmCodegenBackend>) -> &'static str {
    if target_is_apple(cgcx) {
        "__LLVM,__bitcode\0"
    } else if target_is_aix(cgcx) {
        ".ipa\0"
    } else {
        ".llvmbc\0"
    }
}

/// Embed the bitcode of an LLVM module in the LLVM module itself.
///
/// This is done primarily for iOS where it appears to be standard to compile C
/// code at least with `-fembed-bitcode` which creates two sections in the
/// executable:
///
/// * __LLVM,__bitcode
/// * __LLVM,__cmdline
///
/// It appears *both* of these sections are necessary to get the linker to
/// recognize what's going on. A suitable cmdline value is taken from the
/// target spec.
///
/// Furthermore debug/O1 builds don't actually embed bitcode but rather just
/// embed an empty section.
///
/// Basically all of this is us attempting to follow in the footsteps of clang
/// on iOS. See #35968 for lots more info.
unsafe fn embed_bitcode(
    cgcx: &CodegenContext<LlvmCodegenBackend>,
    llcx: &llvm::Context,
    llmod: &llvm::Module,
    cmdline: &str,
    bitcode: &[u8],
) {
    // We're adding custom sections to the output object file, but we definitely
    // do not want these custom sections to make their way into the final linked
    // executable. The purpose of these custom sections is for tooling
    // surrounding object files to work with the LLVM IR, if necessary. For
    // example rustc's own LTO will look for LLVM IR inside of the object file
    // in these sections by default.
    //
    // To handle this is a bit different depending on the object file format
    // used by the backend, broken down into a few different categories:
    //
    // * Mach-O - this is for macOS. Inspecting the source code for the native
    //   linker here shows that the `.llvmbc` and `.llvmcmd` sections are
    //   automatically skipped by the linker. In that case there's nothing extra
    //   that we need to do here.
    //
    // * Wasm - the native LLD linker is hard-coded to skip `.llvmbc` and
    //   `.llvmcmd` sections, so there's nothing extra we need to do.
    //
    // * COFF - if we don't do anything the linker will by default copy all
    //   these sections to the output artifact, not what we want! To subvert
    //   this we want to flag the sections we inserted here as
    //   `IMAGE_SCN_LNK_REMOVE`.
    //
    // * ELF - this is very similar to COFF above. One difference is that these
    //   sections are removed from the output linked artifact when
    //   `--gc-sections` is passed, which we pass by default. If that flag isn't
    //   passed though then these sections will show up in the final output.
    //   Additionally the flag that we need to set here is `SHF_EXCLUDE`.
    //
    // * XCOFF - AIX linker ignores content in .ipa and .info if no auxiliary
    //   symbol associated with these sections.
    //
    // Unfortunately, LLVM provides no way to set custom section flags. For ELF
    // and COFF we emit the sections using module level inline assembly for that
    // reason (see issue #90326 for historical background).
    let is_aix = target_is_aix(cgcx);
    let is_apple = target_is_apple(cgcx);
    unsafe {
        if is_apple || is_aix || cgcx.opts.target_triple.triple().starts_with("wasm") {
            // We don't need custom section flags, create LLVM globals.
            let llconst = common::bytes_in_context(llcx, bitcode);
            let llglobal = llvm::LLVMAddGlobal(
                llmod,
                common::val_ty(llconst),
                c"rustc.embedded.module".as_ptr().cast(),
            );
            llvm::LLVMSetInitializer(llglobal, llconst);

            let section = bitcode_section_name(cgcx);
            llvm::LLVMSetSection(llglobal, section.as_ptr().cast());
            llvm::LLVMRustSetLinkage(llglobal, llvm::Linkage::PrivateLinkage);
            llvm::LLVMSetGlobalConstant(llglobal, llvm::True);

            let llconst = common::bytes_in_context(llcx, cmdline.as_bytes());
            let llglobal = llvm::LLVMAddGlobal(
                llmod,
                common::val_ty(llconst),
                c"rustc.embedded.cmdline".as_ptr().cast(),
            );
            llvm::LLVMSetInitializer(llglobal, llconst);
            let section = if is_apple {
                c"__LLVM,__cmdline"
            } else if is_aix {
                c".info"
            } else {
                c".llvmcmd"
            };
            llvm::LLVMSetSection(llglobal, section.as_ptr().cast());
            llvm::LLVMRustSetLinkage(llglobal, llvm::Linkage::PrivateLinkage);
        } else {
            // We need custom section flags, so emit module-level inline assembly.
            let section_flags = if cgcx.is_pe_coff { "n" } else { "e" };
            let asm = create_section_with_flags_asm(".llvmbc", section_flags, bitcode);
            llvm::LLVMAppendModuleInlineAsm(llmod, asm.as_ptr().cast(), asm.len());
            let asm = create_section_with_flags_asm(".llvmcmd", section_flags, cmdline.as_bytes());
            llvm::LLVMAppendModuleInlineAsm(llmod, asm.as_ptr().cast(), asm.len());
        }
    }
}

// Create a `__imp_<symbol> = &symbol` global for every public static `symbol`.
// This is required to satisfy `dllimport` references to static data in .rlibs
// when using MSVC linker. We do this only for data, as linker can fix up
// code references on its own.
// See #26591, #27438
fn create_msvc_imps(
    cgcx: &CodegenContext<LlvmCodegenBackend>,
    llcx: &llvm::Context,
    llmod: &llvm::Module,
) {
    if !cgcx.msvc_imps_needed {
        return;
    }
    // The x86 ABI seems to require that leading underscores are added to symbol
    // names, so we need an extra underscore on x86. There's also a leading
    // '\x01' here which disables LLVM's symbol mangling (e.g., no extra
    // underscores added in front).
    let prefix = if cgcx.target_arch == "x86" { "\x01__imp__" } else { "\x01__imp_" };

    unsafe {
        let ptr_ty = Type::ptr_llcx(llcx);
        let globals = base::iter_globals(llmod)
            .filter(|&val| {
                llvm::LLVMRustGetLinkage(val) == llvm::Linkage::ExternalLinkage
                    && llvm::LLVMIsDeclaration(val) == 0
            })
            .filter_map(|val| {
                // Exclude some symbols that we know are not Rust symbols.
                let name = llvm::get_value_name(val);
                if ignored(name) { None } else { Some((val, name)) }
            })
            .map(move |(val, name)| {
                let mut imp_name = prefix.as_bytes().to_vec();
                imp_name.extend(name);
                let imp_name = CString::new(imp_name).unwrap();
                (imp_name, val)
            })
            .collect::<Vec<_>>();

        for (imp_name, val) in globals {
            let imp = llvm::LLVMAddGlobal(llmod, ptr_ty, imp_name.as_ptr().cast());
            llvm::LLVMSetInitializer(imp, val);
            llvm::LLVMRustSetLinkage(imp, llvm::Linkage::ExternalLinkage);
        }
    }

    // Use this function to exclude certain symbols from `__imp` generation.
    fn ignored(symbol_name: &[u8]) -> bool {
        // These are symbols generated by LLVM's profiling instrumentation
        symbol_name.starts_with(b"__llvm_profile_")
    }
}

fn record_artifact_size(
    self_profiler_ref: &SelfProfilerRef,
    artifact_kind: &'static str,
    path: &Path,
) {
    // Don't stat the file if we are not going to record its size.
    if !self_profiler_ref.enabled() {
        return;
    }

    if let Some(artifact_name) = path.file_name() {
        let file_size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        self_profiler_ref.artifact_size(artifact_kind, artifact_name.to_string_lossy(), file_size);
    }
}

fn record_llvm_cgu_instructions_stats(prof: &SelfProfilerRef, llmod: &llvm::Module) {
    if !prof.enabled() {
        return;
    }

    let raw_stats =
        llvm::build_string(|s| unsafe { llvm::LLVMRustModuleInstructionStats(llmod, s) })
            .expect("cannot get module instruction stats");

    #[derive(serde::Deserialize)]
    struct InstructionsStats {
        module: String,
        total: u64,
    }

    let InstructionsStats { module, total } =
        serde_json::from_str(&raw_stats).expect("cannot parse llvm cgu instructions stats");
    prof.artifact_size("cgu_instructions", module, total);
}
