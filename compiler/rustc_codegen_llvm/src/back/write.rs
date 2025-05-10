use std::ffi::{CStr, CString};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::ptr::null_mut;
use std::sync::Arc;
use std::{fs, slice, str};

use libc::{c_char, c_int, c_void, size_t};
use llvm::{
    LLVMRustLLVMHasZlibCompressionForDebugSymbols, LLVMRustLLVMHasZstdCompressionForDebugSymbols,
};
use rustc_codegen_ssa::back::link::ensure_removed;
use rustc_codegen_ssa::back::versioned_llvm_target;
use rustc_codegen_ssa::back::write::{
    BitcodeSection, CodegenContext, EmitObj, ModuleConfig, TargetMachineFactoryConfig,
    TargetMachineFactoryFn,
};
use rustc_codegen_ssa::traits::*;
use rustc_codegen_ssa::{CompiledModule, ModuleCodegen, ModuleKind};
use rustc_data_structures::profiling::SelfProfilerRef;
use rustc_data_structures::small_c_str::SmallCStr;
use rustc_errors::{DiagCtxtHandle, FatalError, Level};
use rustc_fs_util::{link_or_copy, path_to_c_string};
use rustc_middle::ty::TyCtxt;
use rustc_session::Session;
use rustc_session::config::{
    self, Lto, OutputType, Passes, RemapPathScopeComponents, SplitDwarfKind, SwitchWithOptPath,
};
use rustc_span::{BytePos, InnerSpan, Pos, SpanData, SyntaxContext, sym};
use rustc_target::spec::{CodeModel, FloatAbi, RelocModel, SanitizerSet, SplitDebuginfo, TlsModel};
use tracing::{debug, trace};

use crate::back::lto::ThinBuffer;
use crate::back::owned_target_machine::OwnedTargetMachine;
use crate::back::profiling::{
    LlvmSelfProfiler, selfprofile_after_pass_callback, selfprofile_before_pass_callback,
};
use crate::common::AsCCharPtr;
use crate::errors::{
    CopyBitcode, FromLlvmDiag, FromLlvmOptimizationDiag, LlvmError, UnknownCompression,
    WithLlvmError, WriteBytecode,
};
use crate::llvm::diagnostic::OptimizationDiagnosticKind::*;
use crate::llvm::{self, DiagnosticInfo};
use crate::type_::Type;
use crate::{LlvmCodegenBackend, ModuleLlvm, base, common, llvm_util};

pub(crate) fn llvm_err<'a>(dcx: DiagCtxtHandle<'_>, err: LlvmError<'a>) -> FatalError {
    match llvm::last_error() {
        Some(llvm_err) => dcx.emit_almost_fatal(WithLlvmError(err, llvm_err)),
        None => dcx.emit_almost_fatal(err),
    }
}

fn write_output_file<'ll>(
    dcx: DiagCtxtHandle<'_>,
    target: &'ll llvm::TargetMachine,
    no_builtins: bool,
    m: &'ll llvm::Module,
    output: &Path,
    dwo_output: Option<&Path>,
    file_type: llvm::FileType,
    self_profiler_ref: &SelfProfilerRef,
    verify_llvm_ir: bool,
) -> Result<(), FatalError> {
    debug!("write_output_file output={:?} dwo_output={:?}", output, dwo_output);
    let output_c = path_to_c_string(output);
    let dwo_output_c;
    let dwo_output_ptr = if let Some(dwo_output) = dwo_output {
        dwo_output_c = path_to_c_string(dwo_output);
        dwo_output_c.as_ptr()
    } else {
        std::ptr::null()
    };
    let result = unsafe {
        let pm = llvm::LLVMCreatePassManager();
        llvm::LLVMAddAnalysisPasses(target, pm);
        llvm::LLVMRustAddLibraryInfo(pm, m, no_builtins);
        llvm::LLVMRustWriteOutputFile(
            target,
            pm,
            m,
            output_c.as_ptr(),
            dwo_output_ptr,
            file_type,
            verify_llvm_ir,
        )
    };

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

pub(crate) fn create_informational_target_machine(
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

pub(crate) fn create_target_machine(tcx: TyCtxt<'_>, mod_name: &str) -> OwnedTargetMachine {
    let split_dwarf_file = if tcx.sess.target_can_use_split_dwarf() {
        tcx.output_filenames(()).split_dwarf_path(
            tcx.sess.split_debuginfo(),
            tcx.sess.opts.unstable_opts.split_dwarf_kind,
            mod_name,
            tcx.sess.invocation_temp.as_deref(),
        )
    } else {
        None
    };

    let output_obj_file = Some(tcx.output_filenames(()).temp_path_for_cgu(
        OutputType::Object,
        mod_name,
        tcx.sess.invocation_temp.as_deref(),
    ));
    let config = TargetMachineFactoryConfig { split_dwarf_file, output_obj_file };

    target_machine_factory(
        tcx.sess,
        tcx.backend_optimization_level(()),
        tcx.global_backend_features(()),
    )(config)
    .unwrap_or_else(|err| llvm_err(tcx.dcx(), err).raise())
}

fn to_llvm_opt_settings(cfg: config::OptLevel) -> (llvm::CodeGenOptLevel, llvm::CodeGenOptSize) {
    use self::config::OptLevel::*;
    match cfg {
        No => (llvm::CodeGenOptLevel::None, llvm::CodeGenOptSizeNone),
        Less => (llvm::CodeGenOptLevel::Less, llvm::CodeGenOptSizeNone),
        More => (llvm::CodeGenOptLevel::Default, llvm::CodeGenOptSizeNone),
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
        More => llvm::PassBuilderOptLevel::O2,
        Aggressive => llvm::PassBuilderOptLevel::O3,
        Size => llvm::PassBuilderOptLevel::Os,
        SizeMin => llvm::PassBuilderOptLevel::Oz,
    }
}

fn to_llvm_relocation_model(relocation_model: RelocModel) -> llvm::RelocModel {
    match relocation_model {
        RelocModel::Static => llvm::RelocModel::Static,
        // LLVM doesn't have a PIE relocation model, it represents PIE as PIC with an extra
        // attribute.
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

fn to_llvm_float_abi(float_abi: Option<FloatAbi>) -> llvm::FloatAbi {
    match float_abi {
        None => llvm::FloatAbi::Default,
        Some(FloatAbi::Soft) => llvm::FloatAbi::Soft,
        Some(FloatAbi::Hard) => llvm::FloatAbi::Hard,
    }
}

pub(crate) fn target_machine_factory(
    sess: &Session,
    optlvl: config::OptLevel,
    target_features: &[String],
) -> TargetMachineFactoryFn<LlvmCodegenBackend> {
    let reloc_model = to_llvm_relocation_model(sess.relocation_model());

    let (opt_level, _) = to_llvm_opt_settings(optlvl);
    let float_abi = if sess.target.arch == "arm" && sess.opts.cg.soft_float {
        llvm::FloatAbi::Soft
    } else {
        // `validate_commandline_args_with_session_available` has already warned about this being
        // ignored. Let's make sure LLVM doesn't suddenly start using this flag on more targets.
        to_llvm_float_abi(sess.target.llvm_floatabi)
    };

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

    let triple = SmallCStr::new(&versioned_llvm_target(sess));
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
            float_abi,
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
    let ext = format!("{name}.bc");
    let path = cgcx.output_filenames.temp_path_ext_for_cgu(
        &ext,
        &module.name,
        cgcx.invocation_temp.as_deref(),
    );
    write_bitcode_to_file(module, &path)
}

fn write_bitcode_to_file(module: &ModuleCodegen<ModuleLlvm>, path: &Path) {
    unsafe {
        let path = path_to_c_string(&path);
        let llmod = module.module_llvm.llmod();
        llvm::LLVMWriteBitcodeToFile(llmod, path.as_ptr());
    }
}

/// In what context is a dignostic handler being attached to a codegen unit?
pub(crate) enum CodegenDiagnosticsStage {
    /// Prelink optimization stage.
    Opt,
    /// LTO/ThinLTO postlink optimization stage.
    LTO,
    /// Code generation.
    Codegen,
}

pub(crate) struct DiagnosticHandlers<'a> {
    data: *mut (&'a CodegenContext<LlvmCodegenBackend>, DiagCtxtHandle<'a>),
    llcx: &'a llvm::Context,
    old_handler: Option<&'a llvm::DiagnosticHandler>,
}

impl<'a> DiagnosticHandlers<'a> {
    pub(crate) fn new(
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
    cookie: u64,
    source: Option<(String, Vec<InnerSpan>)>,
) {
    // In LTO build we may get srcloc values from other crates which are invalid
    // since they use a different source map. To be safe we just suppress these
    // in LTO builds.
    let span = if cookie == 0 || matches!(cgcx.lto, Lto::Fat | Lto::Thin) {
        SpanData::default()
    } else {
        SpanData {
            lo: BytePos::from_u32(cookie as u32),
            hi: BytePos::from_u32((cookie >> 32) as u32),
            ctxt: SyntaxContext::root(),
            parent: None,
        }
    };
    let level = match level {
        llvm::DiagnosticLevel::Error => Level::Error,
        llvm::DiagnosticLevel::Warning => Level::Warning,
        llvm::DiagnosticLevel::Note | llvm::DiagnosticLevel::Remark => Level::Note,
    };
    let msg = msg.strip_prefix("error: ").unwrap_or(&msg).to_string();
    cgcx.diag_emitter.inline_asm_error(span, msg, level, source);
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
                    OptimizationRemark => "success",
                    OptimizationMissed | OptimizationFailure => "missed",
                    OptimizationAnalysis
                    | OptimizationAnalysisFPCommute
                    | OptimizationAnalysisAliasing => "analysis",
                    OptimizationRemarkOther => "other",
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
    config.instrument_coverage.then(|| c"default_%m_%p.profraw".to_owned())
}

// PreAD will run llvm opts but disable size increasing opts (vectorization, loop unrolling)
// DuringAD is the same as above, but also runs the enzyme opt and autodiff passes.
// PostAD will run all opts, including size increasing opts.
#[derive(Debug, Eq, PartialEq)]
pub(crate) enum AutodiffStage {
    PreAD,
    DuringAD,
    PostAD,
}

pub(crate) unsafe fn llvm_optimize(
    cgcx: &CodegenContext<LlvmCodegenBackend>,
    dcx: DiagCtxtHandle<'_>,
    module: &ModuleCodegen<ModuleLlvm>,
    thin_lto_buffer: Option<&mut *mut llvm::ThinLTOBuffer>,
    config: &ModuleConfig,
    opt_level: config::OptLevel,
    opt_stage: llvm::OptStage,
    autodiff_stage: AutodiffStage,
) -> Result<(), FatalError> {
    // Enzyme:
    // The whole point of compiler based AD is to differentiate optimized IR instead of unoptimized
    // source code. However, benchmarks show that optimizations increasing the code size
    // tend to reduce AD performance. Therefore deactivate them before AD, then differentiate the code
    // and finally re-optimize the module, now with all optimizations available.
    // FIXME(ZuseZ4): In a future update we could figure out how to only optimize individual functions getting
    // differentiated.

    let consider_ad = cfg!(llvm_enzyme) && config.autodiff.contains(&config::AutoDiff::Enable);
    let run_enzyme = autodiff_stage == AutodiffStage::DuringAD;
    let print_before_enzyme = config.autodiff.contains(&config::AutoDiff::PrintModBefore);
    let print_after_enzyme = config.autodiff.contains(&config::AutoDiff::PrintModAfter);
    let print_passes = config.autodiff.contains(&config::AutoDiff::PrintPasses);
    let merge_functions;
    let unroll_loops;
    let vectorize_slp;
    let vectorize_loop;

    // When we build rustc with enzyme/autodiff support, we want to postpone size-increasing
    // optimizations until after differentiation. Our pipeline is thus: (opt + enzyme), (full opt).
    // We therefore have two calls to llvm_optimize, if autodiff is used.
    //
    // We also must disable merge_functions, since autodiff placeholder/dummy bodies tend to be
    // identical. We run opts before AD, so there is a chance that LLVM will merge our dummies.
    // In that case, we lack some dummy bodies and can't replace them with the real AD code anymore.
    // We then would need to abort compilation. This was especially common in test cases.
    if consider_ad && autodiff_stage != AutodiffStage::PostAD {
        merge_functions = false;
        unroll_loops = false;
        vectorize_slp = false;
        vectorize_loop = false;
    } else {
        unroll_loops =
            opt_level != config::OptLevel::Size && opt_level != config::OptLevel::SizeMin;
        merge_functions = config.merge_functions;
        vectorize_slp = config.vectorize_slp;
        vectorize_loop = config.vectorize_loop;
    }
    trace!(?unroll_loops, ?vectorize_slp, ?vectorize_loop, ?run_enzyme);
    if thin_lto_buffer.is_some() {
        assert!(
            matches!(
                opt_stage,
                llvm::OptStage::PreLinkNoLTO
                    | llvm::OptStage::PreLinkFatLTO
                    | llvm::OptStage::PreLinkThinLTO
            ),
            "the bitcode for LTO can only be obtained at the pre-link stage"
        );
    }
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
            &*module.module_llvm.tm.raw(),
            to_pass_builder_opt_level(opt_level),
            opt_stage,
            cgcx.opts.cg.linker_plugin_lto.enabled(),
            config.no_prepopulate_passes,
            config.verify_llvm_ir,
            config.lint_llvm_ir,
            thin_lto_buffer,
            config.emit_thin_lto,
            config.emit_thin_lto_summary,
            merge_functions,
            unroll_loops,
            vectorize_slp,
            vectorize_loop,
            config.no_builtins,
            config.emit_lifetime_markers,
            run_enzyme,
            print_before_enzyme,
            print_after_enzyme,
            print_passes,
            sanitizer_options.as_ref(),
            pgo_gen_path.as_ref().map_or(std::ptr::null(), |s| s.as_ptr()),
            pgo_use_path.as_ref().map_or(std::ptr::null(), |s| s.as_ptr()),
            config.instrument_coverage,
            instr_profile_output_path.as_ref().map_or(std::ptr::null(), |s| s.as_ptr()),
            pgo_sample_use_path.as_ref().map_or(std::ptr::null(), |s| s.as_ptr()),
            config.debug_info_for_profiling,
            llvm_selfprofiler,
            selfprofile_before_pass_callback,
            selfprofile_after_pass_callback,
            extra_passes.as_c_char_ptr(),
            extra_passes.len(),
            llvm_plugins.as_c_char_ptr(),
            llvm_plugins.len(),
        )
    };
    result.into_result().map_err(|()| llvm_err(dcx, LlvmError::RunLlvmPasses))
}

// Unsafe due to LLVM calls.
pub(crate) unsafe fn optimize(
    cgcx: &CodegenContext<LlvmCodegenBackend>,
    dcx: DiagCtxtHandle<'_>,
    module: &mut ModuleCodegen<ModuleLlvm>,
    config: &ModuleConfig,
) -> Result<(), FatalError> {
    let _timer = cgcx.prof.generic_activity_with_arg("LLVM_module_optimize", &*module.name);

    let llcx = &*module.module_llvm.llcx;
    let _handlers = DiagnosticHandlers::new(cgcx, dcx, llcx, module, CodegenDiagnosticsStage::Opt);

    if config.emit_no_opt_bc {
        let out = cgcx.output_filenames.temp_path_ext_for_cgu(
            "no-opt.bc",
            &module.name,
            cgcx.invocation_temp.as_deref(),
        );
        write_bitcode_to_file(module, &out)
    }

    // FIXME(ZuseZ4): support SanitizeHWAddress and prevent illegal/unsupported opts

    if let Some(opt_level) = config.opt_level {
        let opt_stage = match cgcx.lto {
            Lto::Fat => llvm::OptStage::PreLinkFatLTO,
            Lto::Thin | Lto::ThinLocal => llvm::OptStage::PreLinkThinLTO,
            _ if cgcx.opts.cg.linker_plugin_lto.enabled() => llvm::OptStage::PreLinkThinLTO,
            _ => llvm::OptStage::PreLinkNoLTO,
        };

        // If we know that we will later run AD, then we disable vectorization and loop unrolling.
        // Otherwise we pretend AD is already done and run the normal opt pipeline (=PostAD).
        let consider_ad = cfg!(llvm_enzyme) && config.autodiff.contains(&config::AutoDiff::Enable);
        let autodiff_stage = if consider_ad { AutodiffStage::PreAD } else { AutodiffStage::PostAD };
        // The embedded bitcode is used to run LTO/ThinLTO.
        // The bitcode obtained during the `codegen` phase is no longer suitable for performing LTO.
        // It may have undergone LTO due to ThinLocal, so we need to obtain the embedded bitcode at
        // this point.
        let mut thin_lto_buffer = if (module.kind == ModuleKind::Regular
            && config.emit_obj == EmitObj::ObjectCode(BitcodeSection::Full))
            || config.emit_thin_lto_summary
        {
            Some(null_mut())
        } else {
            None
        };
        unsafe {
            llvm_optimize(
                cgcx,
                dcx,
                module,
                thin_lto_buffer.as_mut(),
                config,
                opt_level,
                opt_stage,
                autodiff_stage,
            )
        }?;
        if let Some(thin_lto_buffer) = thin_lto_buffer {
            let thin_lto_buffer = unsafe { ThinBuffer::from_raw_ptr(thin_lto_buffer) };
            module.thin_lto_buffer = Some(thin_lto_buffer.data().to_vec());
            let bc_summary_out = cgcx.output_filenames.temp_path_for_cgu(
                OutputType::ThinLinkBitcode,
                &module.name,
                cgcx.invocation_temp.as_deref(),
            );
            if config.emit_thin_lto_summary
                && let Some(thin_link_bitcode_filename) = bc_summary_out.file_name()
            {
                let summary_data = thin_lto_buffer.thin_link_data();
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
        }
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
        let _handlers =
            DiagnosticHandlers::new(cgcx, dcx, llcx, &module, CodegenDiagnosticsStage::Codegen);

        if cgcx.msvc_imps_needed {
            create_msvc_imps(cgcx, llcx, llmod);
        }

        // Note that if object files are just LLVM bitcode we write bitcode,
        // copy it to the .o file, and delete the bitcode if it wasn't
        // otherwise requested.

        let bc_out = cgcx.output_filenames.temp_path_for_cgu(
            OutputType::Bitcode,
            &module.name,
            cgcx.invocation_temp.as_deref(),
        );
        let obj_out = cgcx.output_filenames.temp_path_for_cgu(
            OutputType::Object,
            &module.name,
            cgcx.invocation_temp.as_deref(),
        );

        if config.bitcode_needed() {
            if config.emit_bc || config.emit_obj == EmitObj::Bitcode {
                let thin = {
                    let _timer = cgcx.prof.generic_activity_with_arg(
                        "LLVM_module_codegen_make_bitcode",
                        &*module.name,
                    );
                    ThinBuffer::new(llmod, config.emit_thin_lto, false)
                };
                let data = thin.data();
                let _timer = cgcx
                    .prof
                    .generic_activity_with_arg("LLVM_module_codegen_emit_bitcode", &*module.name);
                if let Some(bitcode_filename) = bc_out.file_name() {
                    cgcx.prof.artifact_size(
                        "llvm_bitcode",
                        bitcode_filename.to_string_lossy(),
                        data.len() as u64,
                    );
                }
                if let Err(err) = fs::write(&bc_out, data) {
                    dcx.emit_err(WriteBytecode { path: &bc_out, err });
                }
            }

            if config.embed_bitcode() && module.kind == ModuleKind::Regular {
                let _timer = cgcx
                    .prof
                    .generic_activity_with_arg("LLVM_module_codegen_embed_bitcode", &*module.name);
                let thin_bc =
                    module.thin_lto_buffer.as_deref().expect("cannot find embedded bitcode");
                unsafe {
                    embed_bitcode(cgcx, llcx, llmod, &config.bc_cmdline, &thin_bc);
                }
            }
        }

        if config.emit_ir {
            let _timer =
                cgcx.prof.generic_activity_with_arg("LLVM_module_codegen_emit_ir", &*module.name);
            let out = cgcx.output_filenames.temp_path_for_cgu(
                OutputType::LlvmAssembly,
                &module.name,
                cgcx.invocation_temp.as_deref(),
            );
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
            let path = cgcx.output_filenames.temp_path_for_cgu(
                OutputType::Assembly,
                &module.name,
                cgcx.invocation_temp.as_deref(),
            );

            // We can't use the same module for asm and object code output,
            // because that triggers various errors like invalid IR or broken
            // binaries. So we must clone the module to produce the asm output
            // if we are also producing object code.
            let llmod = if let EmitObj::ObjectCode(_) = config.emit_obj {
                unsafe { llvm::LLVMCloneModule(llmod) }
            } else {
                llmod
            };
            write_output_file(
                dcx,
                tm.raw(),
                config.no_builtins,
                llmod,
                &path,
                None,
                llvm::FileType::AssemblyFile,
                &cgcx.prof,
                config.verify_llvm_ir,
            )?;
        }

        match config.emit_obj {
            EmitObj::ObjectCode(_) => {
                let _timer = cgcx
                    .prof
                    .generic_activity_with_arg("LLVM_module_codegen_emit_obj", &*module.name);

                let dwo_out = cgcx
                    .output_filenames
                    .temp_path_dwo_for_cgu(&module.name, cgcx.invocation_temp.as_deref());
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

                write_output_file(
                    dcx,
                    tm.raw(),
                    config.no_builtins,
                    llmod,
                    &obj_out,
                    dwo_out,
                    llvm::FileType::ObjectFile,
                    &cgcx.prof,
                    config.verify_llvm_ir,
                )?;
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
        cgcx.invocation_temp.as_deref(),
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

pub(crate) fn bitcode_section_name(cgcx: &CodegenContext<LlvmCodegenBackend>) -> &'static CStr {
    if cgcx.target_is_like_darwin {
        c"__LLVM,__bitcode"
    } else if cgcx.target_is_like_aix {
        c".ipa"
    } else {
        c".llvmbc"
    }
}

/// Embed the bitcode of an LLVM module for LTO in the LLVM module itself.
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
    unsafe {
        if cgcx.target_is_like_darwin
            || cgcx.target_is_like_aix
            || cgcx.target_arch == "wasm32"
            || cgcx.target_arch == "wasm64"
        {
            // We don't need custom section flags, create LLVM globals.
            let llconst = common::bytes_in_context(llcx, bitcode);
            let llglobal =
                llvm::add_global(llmod, common::val_ty(llconst), c"rustc.embedded.module");
            llvm::set_initializer(llglobal, llconst);

            llvm::set_section(llglobal, bitcode_section_name(cgcx));
            llvm::set_linkage(llglobal, llvm::Linkage::PrivateLinkage);
            llvm::LLVMSetGlobalConstant(llglobal, llvm::True);

            let llconst = common::bytes_in_context(llcx, cmdline.as_bytes());
            let llglobal =
                llvm::add_global(llmod, common::val_ty(llconst), c"rustc.embedded.cmdline");
            llvm::set_initializer(llglobal, llconst);
            let section = if cgcx.target_is_like_darwin {
                c"__LLVM,__cmdline"
            } else if cgcx.target_is_like_aix {
                c".info"
            } else {
                c".llvmcmd"
            };
            llvm::set_section(llglobal, section);
            llvm::set_linkage(llglobal, llvm::Linkage::PrivateLinkage);
        } else {
            // We need custom section flags, so emit module-level inline assembly.
            let section_flags = if cgcx.is_pe_coff { "n" } else { "e" };
            let asm = create_section_with_flags_asm(".llvmbc", section_flags, bitcode);
            llvm::append_module_inline_asm(llmod, &asm);
            let asm = create_section_with_flags_asm(".llvmcmd", section_flags, cmdline.as_bytes());
            llvm::append_module_inline_asm(llmod, &asm);
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

    let ptr_ty = Type::ptr_llcx(llcx);
    let globals = base::iter_globals(llmod)
        .filter(|&val| {
            llvm::get_linkage(val) == llvm::Linkage::ExternalLinkage && !llvm::is_declaration(val)
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
        let imp = llvm::add_global(llmod, ptr_ty, &imp_name);

        llvm::set_initializer(imp, val);
        llvm::set_linkage(imp, llvm::Linkage::ExternalLinkage);
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
