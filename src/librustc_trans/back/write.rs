// Copyright 2013-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use back::bytecode::{self, RLIB_BYTECODE_EXTENSION};
use back::lto::{self, ModuleBuffer, ThinBuffer};
use back::link::{self, get_linker, remove};
use back::linker::LinkerInfo;
use back::symbol_export::ExportedSymbols;
use base;
use consts;
use rustc_incremental::{save_trans_partition, in_incr_comp_dir};
use rustc::dep_graph::{DepGraph, WorkProductFileKind};
use rustc::middle::cstore::{LinkMeta, EncodedMetadata};
use rustc::session::config::{self, OutputFilenames, OutputType, OutputTypes, Passes, SomePasses,
                             AllPasses, Sanitizer};
use rustc::session::Session;
use rustc::util::nodemap::FxHashMap;
use rustc_back::LinkerFlavor;
use time_graph::{self, TimeGraph, Timeline};
use llvm;
use llvm::{ModuleRef, TargetMachineRef, PassManagerRef, DiagnosticInfoRef};
use llvm::{SMDiagnosticRef, ContextRef};
use {CrateTranslation, ModuleSource, ModuleTranslation, CompiledModule, ModuleKind};
use CrateInfo;
use rustc::hir::def_id::{CrateNum, LOCAL_CRATE};
use rustc::ty::TyCtxt;
use rustc::util::common::{time, time_depth, set_time_depth, path2cstr, print_time_passes_entry};
use rustc::util::fs::{link_or_copy, rename_or_copy_remove};
use errors::{self, Handler, Level, DiagnosticBuilder, FatalError, DiagnosticId};
use errors::emitter::{Emitter};
use syntax::attr;
use syntax::ext::hygiene::Mark;
use syntax_pos::MultiSpan;
use syntax_pos::symbol::Symbol;
use type_::Type;
use context::{is_pie_binary, get_reloc_model};
use jobserver::{Client, Acquired};
use rustc_demangle;

use std::any::Any;
use std::ffi::{CString, CStr};
use std::fs;
use std::io::{self, Write};
use std::mem;
use std::path::{Path, PathBuf};
use std::str;
use std::sync::Arc;
use std::sync::mpsc::{channel, Sender, Receiver};
use std::slice;
use std::time::Instant;
use std::thread;
use libc::{c_uint, c_void, c_char, size_t};

pub const RELOC_MODEL_ARGS : [(&'static str, llvm::RelocMode); 7] = [
    ("pic", llvm::RelocMode::PIC),
    ("static", llvm::RelocMode::Static),
    ("default", llvm::RelocMode::Default),
    ("dynamic-no-pic", llvm::RelocMode::DynamicNoPic),
    ("ropi", llvm::RelocMode::ROPI),
    ("rwpi", llvm::RelocMode::RWPI),
    ("ropi-rwpi", llvm::RelocMode::ROPI_RWPI),
];

pub const CODE_GEN_MODEL_ARGS : [(&'static str, llvm::CodeModel); 5] = [
    ("default", llvm::CodeModel::Default),
    ("small", llvm::CodeModel::Small),
    ("kernel", llvm::CodeModel::Kernel),
    ("medium", llvm::CodeModel::Medium),
    ("large", llvm::CodeModel::Large),
];

pub const TLS_MODEL_ARGS : [(&'static str, llvm::ThreadLocalMode); 4] = [
    ("global-dynamic", llvm::ThreadLocalMode::GeneralDynamic),
    ("local-dynamic", llvm::ThreadLocalMode::LocalDynamic),
    ("initial-exec", llvm::ThreadLocalMode::InitialExec),
    ("local-exec", llvm::ThreadLocalMode::LocalExec),
];

pub fn llvm_err(handler: &errors::Handler, msg: String) -> FatalError {
    match llvm::last_error() {
        Some(err) => handler.fatal(&format!("{}: {}", msg, err)),
        None => handler.fatal(&msg),
    }
}

pub fn write_output_file(
        handler: &errors::Handler,
        target: llvm::TargetMachineRef,
        pm: llvm::PassManagerRef,
        m: ModuleRef,
        output: &Path,
        file_type: llvm::FileType) -> Result<(), FatalError> {
    unsafe {
        let output_c = path2cstr(output);
        let result = llvm::LLVMRustWriteOutputFile(
                target, pm, m, output_c.as_ptr(), file_type);
        if result.into_result().is_err() {
            let msg = format!("could not write output to {}", output.display());
            Err(llvm_err(handler, msg))
        } else {
            Ok(())
        }
    }
}

// On android, we by default compile for armv7 processors. This enables
// things like double word CAS instructions (rather than emulating them)
// which are *far* more efficient. This is obviously undesirable in some
// cases, so if any sort of target feature is specified we don't append v7
// to the feature list.
//
// On iOS only armv7 and newer are supported. So it is useful to
// get all hardware potential via VFP3 (hardware floating point)
// and NEON (SIMD) instructions supported by LLVM.
// Note that without those flags various linking errors might
// arise as some of intrinsics are converted into function calls
// and nobody provides implementations those functions
fn target_feature(sess: &Session) -> String {
    let rustc_features = [
        "crt-static",
    ];
    let requested_features = sess.opts.cg.target_feature.split(',');
    let llvm_features = requested_features.filter(|f| {
        !rustc_features.iter().any(|s| f.contains(s))
    });
    format!("{},{}",
            sess.target.target.options.features,
            llvm_features.collect::<Vec<_>>().join(","))
}

fn get_llvm_opt_level(optimize: config::OptLevel) -> llvm::CodeGenOptLevel {
    match optimize {
      config::OptLevel::No => llvm::CodeGenOptLevel::None,
      config::OptLevel::Less => llvm::CodeGenOptLevel::Less,
      config::OptLevel::Default => llvm::CodeGenOptLevel::Default,
      config::OptLevel::Aggressive => llvm::CodeGenOptLevel::Aggressive,
      _ => llvm::CodeGenOptLevel::Default,
    }
}

fn get_llvm_opt_size(optimize: config::OptLevel) -> llvm::CodeGenOptSize {
    match optimize {
      config::OptLevel::Size => llvm::CodeGenOptSizeDefault,
      config::OptLevel::SizeMin => llvm::CodeGenOptSizeAggressive,
      _ => llvm::CodeGenOptSizeNone,
    }
}

pub fn create_target_machine(sess: &Session) -> TargetMachineRef {
    target_machine_factory(sess)().unwrap_or_else(|err| {
        panic!(llvm_err(sess.diagnostic(), err))
    })
}

pub fn target_machine_factory(sess: &Session)
    -> Arc<Fn() -> Result<TargetMachineRef, String> + Send + Sync>
{
    let reloc_model = get_reloc_model(sess);

    let opt_level = get_llvm_opt_level(sess.opts.optimize);
    let use_softfp = sess.opts.cg.soft_float;

    let ffunction_sections = sess.target.target.options.function_sections;
    let fdata_sections = ffunction_sections;

    let code_model_arg = match sess.opts.cg.code_model {
        Some(ref s) => &s,
        None => &sess.target.target.options.code_model,
    };

    let code_model = match CODE_GEN_MODEL_ARGS.iter().find(
        |&&arg| arg.0 == code_model_arg) {
        Some(x) => x.1,
        _ => {
            sess.err(&format!("{:?} is not a valid code model",
                              code_model_arg));
            sess.abort_if_errors();
            bug!();
        }
    };

    let singlethread = sess.target.target.options.singlethread;

    let triple = &sess.target.target.llvm_target;

    let triple = CString::new(triple.as_bytes()).unwrap();
    let cpu = match sess.opts.cg.target_cpu {
        Some(ref s) => &**s,
        None => &*sess.target.target.options.cpu
    };
    let cpu = CString::new(cpu.as_bytes()).unwrap();
    let features = CString::new(target_feature(sess).as_bytes()).unwrap();
    let is_pie_binary = is_pie_binary(sess);
    let trap_unreachable = sess.target.target.options.trap_unreachable;

    Arc::new(move || {
        let tm = unsafe {
            llvm::LLVMRustCreateTargetMachine(
                triple.as_ptr(), cpu.as_ptr(), features.as_ptr(),
                code_model,
                reloc_model,
                opt_level,
                use_softfp,
                is_pie_binary,
                ffunction_sections,
                fdata_sections,
                trap_unreachable,
                singlethread,
            )
        };

        if tm.is_null() {
            Err(format!("Could not create LLVM TargetMachine for triple: {}",
                        triple.to_str().unwrap()))
        } else {
            Ok(tm)
        }
    })
}

/// Module-specific configuration for `optimize_and_codegen`.
pub struct ModuleConfig {
    /// Names of additional optimization passes to run.
    passes: Vec<String>,
    /// Some(level) to optimize at a certain level, or None to run
    /// absolutely no optimizations (used for the metadata module).
    pub opt_level: Option<llvm::CodeGenOptLevel>,

    /// Some(level) to optimize binary size, or None to not affect program size.
    opt_size: Option<llvm::CodeGenOptSize>,

    // Flags indicating which outputs to produce.
    emit_no_opt_bc: bool,
    emit_bc: bool,
    emit_bc_compressed: bool,
    emit_lto_bc: bool,
    emit_ir: bool,
    emit_asm: bool,
    emit_obj: bool,
    // Miscellaneous flags.  These are mostly copied from command-line
    // options.
    no_verify: bool,
    no_prepopulate_passes: bool,
    no_builtins: bool,
    time_passes: bool,
    vectorize_loop: bool,
    vectorize_slp: bool,
    merge_functions: bool,
    inline_threshold: Option<usize>,
    // Instead of creating an object file by doing LLVM codegen, just
    // make the object file bitcode. Provides easy compatibility with
    // emscripten's ecc compiler, when used as the linker.
    obj_is_bitcode: bool,
}

impl ModuleConfig {
    fn new(passes: Vec<String>) -> ModuleConfig {
        ModuleConfig {
            passes,
            opt_level: None,
            opt_size: None,

            emit_no_opt_bc: false,
            emit_bc: false,
            emit_bc_compressed: false,
            emit_lto_bc: false,
            emit_ir: false,
            emit_asm: false,
            emit_obj: false,
            obj_is_bitcode: false,

            no_verify: false,
            no_prepopulate_passes: false,
            no_builtins: false,
            time_passes: false,
            vectorize_loop: false,
            vectorize_slp: false,
            merge_functions: false,
            inline_threshold: None
        }
    }

    fn set_flags(&mut self, sess: &Session, no_builtins: bool) {
        self.no_verify = sess.no_verify();
        self.no_prepopulate_passes = sess.opts.cg.no_prepopulate_passes;
        self.no_builtins = no_builtins || sess.target.target.options.no_builtins;
        self.time_passes = sess.time_passes();
        self.inline_threshold = sess.opts.cg.inline_threshold;
        self.obj_is_bitcode = sess.target.target.options.obj_is_bitcode;

        // Copy what clang does by turning on loop vectorization at O2 and
        // slp vectorization at O3. Otherwise configure other optimization aspects
        // of this pass manager builder.
        // Turn off vectorization for emscripten, as it's not very well supported.
        self.vectorize_loop = !sess.opts.cg.no_vectorize_loops &&
                             (sess.opts.optimize == config::OptLevel::Default ||
                              sess.opts.optimize == config::OptLevel::Aggressive) &&
                             !sess.target.target.options.is_like_emscripten;

        self.vectorize_slp = !sess.opts.cg.no_vectorize_slp &&
                            sess.opts.optimize == config::OptLevel::Aggressive &&
                            !sess.target.target.options.is_like_emscripten;

        self.merge_functions = sess.opts.optimize == config::OptLevel::Default ||
                               sess.opts.optimize == config::OptLevel::Aggressive;
    }
}

/// Additional resources used by optimize_and_codegen (not module specific)
#[derive(Clone)]
pub struct CodegenContext {
    // Resouces needed when running LTO
    pub time_passes: bool,
    pub lto: bool,
    pub thinlto: bool,
    pub no_landing_pads: bool,
    pub save_temps: bool,
    pub fewer_names: bool,
    pub exported_symbols: Arc<ExportedSymbols>,
    pub opts: Arc<config::Options>,
    pub crate_types: Vec<config::CrateType>,
    pub each_linked_rlib_for_lto: Vec<(CrateNum, PathBuf)>,
    output_filenames: Arc<OutputFilenames>,
    regular_module_config: Arc<ModuleConfig>,
    metadata_module_config: Arc<ModuleConfig>,
    allocator_module_config: Arc<ModuleConfig>,
    pub tm_factory: Arc<Fn() -> Result<TargetMachineRef, String> + Send + Sync>,
    pub msvc_imps_needed: bool,
    pub target_pointer_width: String,
    binaryen_linker: bool,
    debuginfo: config::DebugInfoLevel,
    wasm_import_memory: bool,

    // Number of cgus excluding the allocator/metadata modules
    pub total_cgus: usize,
    // Handler to use for diagnostics produced during codegen.
    pub diag_emitter: SharedEmitter,
    // LLVM passes added by plugins.
    pub plugin_passes: Vec<String>,
    // LLVM optimizations for which we want to print remarks.
    pub remark: Passes,
    // Worker thread number
    pub worker: usize,
    // The incremental compilation session directory, or None if we are not
    // compiling incrementally
    pub incr_comp_session_dir: Option<PathBuf>,
    // Channel back to the main control thread to send messages to
    coordinator_send: Sender<Box<Any + Send>>,
    // A reference to the TimeGraph so we can register timings. None means that
    // measuring is disabled.
    time_graph: Option<TimeGraph>,
}

impl CodegenContext {
    pub fn create_diag_handler(&self) -> Handler {
        Handler::with_emitter(true, false, Box::new(self.diag_emitter.clone()))
    }

    pub fn config(&self, kind: ModuleKind) -> &ModuleConfig {
        match kind {
            ModuleKind::Regular => &self.regular_module_config,
            ModuleKind::Metadata => &self.metadata_module_config,
            ModuleKind::Allocator => &self.allocator_module_config,
        }
    }

    pub fn save_temp_bitcode(&self, trans: &ModuleTranslation, name: &str) {
        if !self.save_temps {
            return
        }
        unsafe {
            let ext = format!("{}.bc", name);
            let cgu = Some(&trans.name[..]);
            let path = self.output_filenames.temp_path_ext(&ext, cgu);
            let cstr = path2cstr(&path);
            let llmod = trans.llvm().unwrap().llmod;
            llvm::LLVMWriteBitcodeToFile(llmod, cstr.as_ptr());
        }
    }
}

struct DiagnosticHandlers<'a> {
    inner: Box<(&'a CodegenContext, &'a Handler)>,
    llcx: ContextRef,
}

impl<'a> DiagnosticHandlers<'a> {
    fn new(cgcx: &'a CodegenContext,
           handler: &'a Handler,
           llcx: ContextRef) -> DiagnosticHandlers<'a> {
        let data = Box::new((cgcx, handler));
        unsafe {
            let arg = &*data as &(_, _) as *const _ as *mut _;
            llvm::LLVMRustSetInlineAsmDiagnosticHandler(llcx, inline_asm_handler, arg);
            llvm::LLVMContextSetDiagnosticHandler(llcx, diagnostic_handler, arg);
        }
        DiagnosticHandlers {
            inner: data,
            llcx: llcx,
        }
    }
}

impl<'a> Drop for DiagnosticHandlers<'a> {
    fn drop(&mut self) {
        unsafe {
            llvm::LLVMRustSetInlineAsmDiagnosticHandler(self.llcx, inline_asm_handler, 0 as *mut _);
            llvm::LLVMContextSetDiagnosticHandler(self.llcx, diagnostic_handler, 0 as *mut _);
        }
    }
}

unsafe extern "C" fn report_inline_asm<'a, 'b>(cgcx: &'a CodegenContext,
                                               msg: &'b str,
                                               cookie: c_uint) {
    cgcx.diag_emitter.inline_asm_error(cookie as u32, msg.to_string());
}

unsafe extern "C" fn inline_asm_handler(diag: SMDiagnosticRef,
                                        user: *const c_void,
                                        cookie: c_uint) {
    if user.is_null() {
        return
    }
    let (cgcx, _) = *(user as *const (&CodegenContext, &Handler));

    let msg = llvm::build_string(|s| llvm::LLVMRustWriteSMDiagnosticToString(diag, s))
        .expect("non-UTF8 SMDiagnostic");

    report_inline_asm(cgcx, &msg, cookie);
}

unsafe extern "C" fn diagnostic_handler(info: DiagnosticInfoRef, user: *mut c_void) {
    if user.is_null() {
        return
    }
    let (cgcx, diag_handler) = *(user as *const (&CodegenContext, &Handler));

    match llvm::diagnostic::Diagnostic::unpack(info) {
        llvm::diagnostic::InlineAsm(inline) => {
            report_inline_asm(cgcx,
                              &llvm::twine_to_string(inline.message),
                              inline.cookie);
        }

        llvm::diagnostic::Optimization(opt) => {
            let enabled = match cgcx.remark {
                AllPasses => true,
                SomePasses(ref v) => v.iter().any(|s| *s == opt.pass_name),
            };

            if enabled {
                diag_handler.note_without_error(&format!("optimization {} for {} at {}:{}:{}: {}",
                                                opt.kind.describe(),
                                                opt.pass_name,
                                                opt.filename,
                                                opt.line,
                                                opt.column,
                                                opt.message));
            }
        }

        _ => (),
    }
}

// Unsafe due to LLVM calls.
unsafe fn optimize(cgcx: &CodegenContext,
                   diag_handler: &Handler,
                   mtrans: &ModuleTranslation,
                   config: &ModuleConfig,
                   timeline: &mut Timeline)
    -> Result<(), FatalError>
{
    let (llmod, llcx, tm) = match mtrans.source {
        ModuleSource::Translated(ref llvm) => (llvm.llmod, llvm.llcx, llvm.tm),
        ModuleSource::Preexisting(_) => {
            bug!("optimize_and_codegen: called with ModuleSource::Preexisting")
        }
    };

    let _handlers = DiagnosticHandlers::new(cgcx, diag_handler, llcx);

    let module_name = mtrans.name.clone();
    let module_name = Some(&module_name[..]);

    if config.emit_no_opt_bc {
        let out = cgcx.output_filenames.temp_path_ext("no-opt.bc", module_name);
        let out = path2cstr(&out);
        llvm::LLVMWriteBitcodeToFile(llmod, out.as_ptr());
    }

    if config.opt_level.is_some() {
        // Create the two optimizing pass managers. These mirror what clang
        // does, and are by populated by LLVM's default PassManagerBuilder.
        // Each manager has a different set of passes, but they also share
        // some common passes.
        let fpm = llvm::LLVMCreateFunctionPassManagerForModule(llmod);
        let mpm = llvm::LLVMCreatePassManager();

        // If we're verifying or linting, add them to the function pass
        // manager.
        let addpass = |pass_name: &str| {
            let pass_name = CString::new(pass_name).unwrap();
            let pass = llvm::LLVMRustFindAndCreatePass(pass_name.as_ptr());
            if pass.is_null() {
                return false;
            }
            let pass_manager = match llvm::LLVMRustPassKind(pass) {
                llvm::PassKind::Function => fpm,
                llvm::PassKind::Module => mpm,
                llvm::PassKind::Other => {
                    diag_handler.err("Encountered LLVM pass kind we can't handle");
                    return true
                },
            };
            llvm::LLVMRustAddPass(pass_manager, pass);
            true
        };

        if !config.no_verify { assert!(addpass("verify")); }
        if !config.no_prepopulate_passes {
            llvm::LLVMRustAddAnalysisPasses(tm, fpm, llmod);
            llvm::LLVMRustAddAnalysisPasses(tm, mpm, llmod);
            let opt_level = config.opt_level.unwrap_or(llvm::CodeGenOptLevel::None);
            with_llvm_pmb(llmod, &config, opt_level, &mut |b| {
                llvm::LLVMPassManagerBuilderPopulateFunctionPassManager(b, fpm);
                llvm::LLVMPassManagerBuilderPopulateModulePassManager(b, mpm);
            })
        }

        for pass in &config.passes {
            if !addpass(pass) {
                diag_handler.warn(&format!("unknown pass `{}`, ignoring",
                                           pass));
            }
        }

        for pass in &cgcx.plugin_passes {
            if !addpass(pass) {
                diag_handler.err(&format!("a plugin asked for LLVM pass \
                                           `{}` but LLVM does not \
                                           recognize it", pass));
            }
        }

        diag_handler.abort_if_errors();

        // Finally, run the actual optimization passes
        time(config.time_passes, &format!("llvm function passes [{}]", module_name.unwrap()), ||
             llvm::LLVMRustRunFunctionPassManager(fpm, llmod));
        timeline.record("fpm");
        time(config.time_passes, &format!("llvm module passes [{}]", module_name.unwrap()), ||
             llvm::LLVMRunPassManager(mpm, llmod));

        // Deallocate managers that we're now done with
        llvm::LLVMDisposePassManager(fpm);
        llvm::LLVMDisposePassManager(mpm);
    }
    Ok(())
}

fn generate_lto_work(cgcx: &CodegenContext,
                     modules: Vec<ModuleTranslation>)
    -> Vec<(WorkItem, u64)>
{
    let mut timeline = cgcx.time_graph.as_ref().map(|tg| {
        tg.start(TRANS_WORKER_TIMELINE,
                 TRANS_WORK_PACKAGE_KIND,
                 "generate lto")
    }).unwrap_or(Timeline::noop());
    let mode = if cgcx.lto {
        lto::LTOMode::WholeCrateGraph
    } else {
        lto::LTOMode::JustThisCrate
    };
    let lto_modules = lto::run(cgcx, modules, mode, &mut timeline)
        .unwrap_or_else(|e| panic!(e));

    lto_modules.into_iter().map(|module| {
        let cost = module.cost();
        (WorkItem::LTO(module), cost)
    }).collect()
}

unsafe fn codegen(cgcx: &CodegenContext,
                  diag_handler: &Handler,
                  mtrans: ModuleTranslation,
                  config: &ModuleConfig,
                  timeline: &mut Timeline)
    -> Result<CompiledModule, FatalError>
{
    timeline.record("codegen");
    let (llmod, llcx, tm) = match mtrans.source {
        ModuleSource::Translated(ref llvm) => (llvm.llmod, llvm.llcx, llvm.tm),
        ModuleSource::Preexisting(_) => {
            bug!("codegen: called with ModuleSource::Preexisting")
        }
    };
    let module_name = mtrans.name.clone();
    let module_name = Some(&module_name[..]);
    let handlers = DiagnosticHandlers::new(cgcx, diag_handler, llcx);

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
    unsafe fn with_codegen<F, R>(tm: TargetMachineRef,
                                 llmod: ModuleRef,
                                 no_builtins: bool,
                                 f: F) -> R
        where F: FnOnce(PassManagerRef) -> R,
    {
        let cpm = llvm::LLVMCreatePassManager();
        llvm::LLVMRustAddAnalysisPasses(tm, cpm, llmod);
        llvm::LLVMRustAddLibraryInfo(cpm, llmod, no_builtins);
        f(cpm)
    }

    // If we're going to generate wasm code from the assembly that llvm
    // generates then we'll be transitively affecting a ton of options below.
    // This only happens on the wasm target now.
    let asm2wasm = cgcx.binaryen_linker &&
        !cgcx.crate_types.contains(&config::CrateTypeRlib) &&
        mtrans.kind == ModuleKind::Regular;

    // Change what we write and cleanup based on whether obj files are
    // just llvm bitcode. In that case write bitcode, and possibly
    // delete the bitcode if it wasn't requested. Don't generate the
    // machine code, instead copy the .o file from the .bc
    let write_bc = config.emit_bc || (config.obj_is_bitcode && !asm2wasm);
    let rm_bc = !config.emit_bc && config.obj_is_bitcode && !asm2wasm;
    let write_obj = config.emit_obj && !config.obj_is_bitcode && !asm2wasm;
    let copy_bc_to_obj = config.emit_obj && config.obj_is_bitcode && !asm2wasm;

    let bc_out = cgcx.output_filenames.temp_path(OutputType::Bitcode, module_name);
    let obj_out = cgcx.output_filenames.temp_path(OutputType::Object, module_name);


    if write_bc || config.emit_bc_compressed {
        let thin;
        let old;
        let data = if llvm::LLVMRustThinLTOAvailable() {
            thin = ThinBuffer::new(llmod);
            thin.data()
        } else {
            old = ModuleBuffer::new(llmod);
            old.data()
        };
        timeline.record("make-bc");

        if write_bc {
            if let Err(e) = fs::write(&bc_out, data) {
                diag_handler.err(&format!("failed to write bytecode: {}", e));
            }
            timeline.record("write-bc");
        }

        if config.emit_bc_compressed {
            let dst = bc_out.with_extension(RLIB_BYTECODE_EXTENSION);
            let data = bytecode::encode(&mtrans.llmod_id, data);
            if let Err(e) = fs::write(&dst, data) {
                diag_handler.err(&format!("failed to write bytecode: {}", e));
            }
            timeline.record("compress-bc");
        }
    }

    time(config.time_passes, &format!("codegen passes [{}]", module_name.unwrap()),
         || -> Result<(), FatalError> {
        if config.emit_ir {
            let out = cgcx.output_filenames.temp_path(OutputType::LlvmAssembly, module_name);
            let out = path2cstr(&out);

            extern "C" fn demangle_callback(input_ptr: *const c_char,
                                            input_len: size_t,
                                            output_ptr: *mut c_char,
                                            output_len: size_t) -> size_t {
                let input = unsafe {
                    slice::from_raw_parts(input_ptr as *const u8, input_len as usize)
                };

                let input = match str::from_utf8(input) {
                    Ok(s) => s,
                    Err(_) => return 0,
                };

                let output = unsafe {
                    slice::from_raw_parts_mut(output_ptr as *mut u8, output_len as usize)
                };
                let mut cursor = io::Cursor::new(output);

                let demangled = match rustc_demangle::try_demangle(input) {
                    Ok(d) => d,
                    Err(_) => return 0,
                };

                if let Err(_) = write!(cursor, "{:#}", demangled) {
                    // Possible only if provided buffer is not big enough
                    return 0;
                }

                cursor.position() as size_t
            }

            with_codegen(tm, llmod, config.no_builtins, |cpm| {
                llvm::LLVMRustPrintModule(cpm, llmod, out.as_ptr(), demangle_callback);
                llvm::LLVMDisposePassManager(cpm);
            });
            timeline.record("ir");
        }

        if config.emit_asm || (asm2wasm && config.emit_obj) {
            let path = cgcx.output_filenames.temp_path(OutputType::Assembly, module_name);

            // We can't use the same module for asm and binary output, because that triggers
            // various errors like invalid IR or broken binaries, so we might have to clone the
            // module to produce the asm output
            let llmod = if config.emit_obj {
                llvm::LLVMCloneModule(llmod)
            } else {
                llmod
            };
            with_codegen(tm, llmod, config.no_builtins, |cpm| {
                write_output_file(diag_handler, tm, cpm, llmod, &path,
                                  llvm::FileType::AssemblyFile)
            })?;
            if config.emit_obj {
                llvm::LLVMDisposeModule(llmod);
            }
            timeline.record("asm");
        }

        if asm2wasm && config.emit_obj {
            let assembly = cgcx.output_filenames.temp_path(OutputType::Assembly, module_name);
            binaryen_assemble(cgcx, diag_handler, &assembly, &obj_out);
            timeline.record("binaryen");

            if !config.emit_asm {
                drop(fs::remove_file(&assembly));
            }
        } else if write_obj {
            with_codegen(tm, llmod, config.no_builtins, |cpm| {
                write_output_file(diag_handler, tm, cpm, llmod, &obj_out,
                                  llvm::FileType::ObjectFile)
            })?;
            timeline.record("obj");
        }

        Ok(())
    })?;

    if copy_bc_to_obj {
        debug!("copying bitcode {:?} to obj {:?}", bc_out, obj_out);
        if let Err(e) = link_or_copy(&bc_out, &obj_out) {
            diag_handler.err(&format!("failed to copy bitcode to object file: {}", e));
        }
    }

    if rm_bc {
        debug!("removing_bitcode {:?}", bc_out);
        if let Err(e) = fs::remove_file(&bc_out) {
            diag_handler.err(&format!("failed to remove bitcode: {}", e));
        }
    }

    drop(handlers);
    Ok(mtrans.into_compiled_module(config.emit_obj,
                                   config.emit_bc,
                                   config.emit_bc_compressed,
                                   &cgcx.output_filenames))
}

/// Translates the LLVM-generated `assembly` on the filesystem into a wasm
/// module using binaryen, placing the output at `object`.
///
/// In this case the "object" is actually a full and complete wasm module. We
/// won't actually be doing anything else to the output for now. This is all
/// pretty janky and will get removed as soon as a linker for wasm exists.
fn binaryen_assemble(cgcx: &CodegenContext,
                     handler: &Handler,
                     assembly: &Path,
                     object: &Path) {
    use rustc_binaryen::{Module, ModuleOptions};

    let input = fs::read(&assembly).and_then(|contents| {
        Ok(CString::new(contents)?)
    });
    let mut options = ModuleOptions::new();
    if cgcx.debuginfo != config::NoDebugInfo {
        options.debuginfo(true);
    }
    if cgcx.crate_types.contains(&config::CrateTypeExecutable) {
        options.start("main");
    }
    options.stack(1024 * 1024);
    options.import_memory(cgcx.wasm_import_memory);
    let assembled = input.and_then(|input| {
        Module::new(&input, &options)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))
    });
    let err = assembled.and_then(|binary| {
        fs::write(&object, binary.data())
    });
    if let Err(e) = err {
        handler.err(&format!("failed to run binaryen assembler: {}", e));
    }
}

pub struct CompiledModules {
    pub modules: Vec<CompiledModule>,
    pub metadata_module: CompiledModule,
    pub allocator_module: Option<CompiledModule>,
}

fn need_crate_bitcode_for_rlib(sess: &Session) -> bool {
    sess.crate_types.borrow().contains(&config::CrateTypeRlib) &&
    sess.opts.output_types.contains_key(&OutputType::Exe)
}

pub fn start_async_translation(tcx: TyCtxt,
                               time_graph: Option<TimeGraph>,
                               link: LinkMeta,
                               metadata: EncodedMetadata,
                               coordinator_receive: Receiver<Box<Any + Send>>,
                               total_cgus: usize)
                               -> OngoingCrateTranslation {
    let sess = tcx.sess;
    let crate_output = tcx.output_filenames(LOCAL_CRATE);
    let crate_name = tcx.crate_name(LOCAL_CRATE);
    let no_builtins = attr::contains_name(&tcx.hir.krate().attrs, "no_builtins");
    let subsystem = attr::first_attr_value_str_by_name(&tcx.hir.krate().attrs,
                                                       "windows_subsystem");
    let windows_subsystem = subsystem.map(|subsystem| {
        if subsystem != "windows" && subsystem != "console" {
            tcx.sess.fatal(&format!("invalid windows subsystem `{}`, only \
                                     `windows` and `console` are allowed",
                                    subsystem));
        }
        subsystem.to_string()
    });

    let no_integrated_as = tcx.sess.opts.cg.no_integrated_as ||
        (tcx.sess.target.target.options.no_integrated_as &&
         (crate_output.outputs.contains_key(&OutputType::Object) ||
          crate_output.outputs.contains_key(&OutputType::Exe)));
    let linker_info = LinkerInfo::new(tcx);
    let crate_info = CrateInfo::new(tcx);

    let output_types_override = if no_integrated_as {
        OutputTypes::new(&[(OutputType::Assembly, None)])
    } else {
        sess.opts.output_types.clone()
    };

    // Figure out what we actually need to build.
    let mut modules_config = ModuleConfig::new(sess.opts.cg.passes.clone());
    let mut metadata_config = ModuleConfig::new(vec![]);
    let mut allocator_config = ModuleConfig::new(vec![]);

    if let Some(ref sanitizer) = sess.opts.debugging_opts.sanitizer {
        match *sanitizer {
            Sanitizer::Address => {
                modules_config.passes.push("asan".to_owned());
                modules_config.passes.push("asan-module".to_owned());
            }
            Sanitizer::Memory => {
                modules_config.passes.push("msan".to_owned())
            }
            Sanitizer::Thread => {
                modules_config.passes.push("tsan".to_owned())
            }
            _ => {}
        }
    }

    if sess.opts.debugging_opts.profile {
        modules_config.passes.push("insert-gcov-profiling".to_owned())
    }

    modules_config.opt_level = Some(get_llvm_opt_level(sess.opts.optimize));
    modules_config.opt_size = Some(get_llvm_opt_size(sess.opts.optimize));

    // Save all versions of the bytecode if we're saving our temporaries.
    if sess.opts.cg.save_temps {
        modules_config.emit_no_opt_bc = true;
        modules_config.emit_bc = true;
        modules_config.emit_lto_bc = true;
        metadata_config.emit_bc = true;
        allocator_config.emit_bc = true;
    }

    // Emit compressed bitcode files for the crate if we're emitting an rlib.
    // Whenever an rlib is created, the bitcode is inserted into the archive in
    // order to allow LTO against it.
    if need_crate_bitcode_for_rlib(sess) {
        modules_config.emit_bc_compressed = true;
        allocator_config.emit_bc_compressed = true;
    }

    for output_type in output_types_override.keys() {
        match *output_type {
            OutputType::Bitcode => { modules_config.emit_bc = true; }
            OutputType::LlvmAssembly => { modules_config.emit_ir = true; }
            OutputType::Assembly => {
                modules_config.emit_asm = true;
                // If we're not using the LLVM assembler, this function
                // could be invoked specially with output_type_assembly, so
                // in this case we still want the metadata object file.
                if !sess.opts.output_types.contains_key(&OutputType::Assembly) {
                    metadata_config.emit_obj = true;
                    allocator_config.emit_obj = true;
                }
            }
            OutputType::Object => { modules_config.emit_obj = true; }
            OutputType::Metadata => { metadata_config.emit_obj = true; }
            OutputType::Exe => {
                modules_config.emit_obj = true;
                metadata_config.emit_obj = true;
                allocator_config.emit_obj = true;
            },
            OutputType::Mir => {}
            OutputType::DepInfo => {}
        }
    }

    modules_config.set_flags(sess, no_builtins);
    metadata_config.set_flags(sess, no_builtins);
    allocator_config.set_flags(sess, no_builtins);

    // Exclude metadata and allocator modules from time_passes output, since
    // they throw off the "LLVM passes" measurement.
    metadata_config.time_passes = false;
    allocator_config.time_passes = false;

    let client = sess.jobserver_from_env.clone().unwrap_or_else(|| {
        // Pick a "reasonable maximum" if we don't otherwise have a jobserver in
        // our environment, capping out at 32 so we don't take everything down
        // by hogging the process run queue.
        Client::new(32).expect("failed to create jobserver")
    });

    let (shared_emitter, shared_emitter_main) = SharedEmitter::new();
    let (trans_worker_send, trans_worker_receive) = channel();

    let coordinator_thread = start_executing_work(tcx,
                                                  &crate_info,
                                                  shared_emitter,
                                                  trans_worker_send,
                                                  coordinator_receive,
                                                  total_cgus,
                                                  client,
                                                  time_graph.clone(),
                                                  Arc::new(modules_config),
                                                  Arc::new(metadata_config),
                                                  Arc::new(allocator_config));

    OngoingCrateTranslation {
        crate_name,
        link,
        metadata,
        windows_subsystem,
        linker_info,
        no_integrated_as,
        crate_info,

        time_graph,
        coordinator_send: tcx.tx_to_llvm_workers.clone(),
        trans_worker_receive,
        shared_emitter_main,
        future: coordinator_thread,
        output_filenames: tcx.output_filenames(LOCAL_CRATE),
    }
}

fn copy_module_artifacts_into_incr_comp_cache(sess: &Session,
                                              dep_graph: &DepGraph,
                                              compiled_modules: &CompiledModules) {
    if sess.opts.incremental.is_none() {
        return;
    }

    for module in compiled_modules.modules.iter() {
        let mut files = vec![];

        if let Some(ref path) = module.object {
            files.push((WorkProductFileKind::Object, path.clone()));
        }
        if let Some(ref path) = module.bytecode {
            files.push((WorkProductFileKind::Bytecode, path.clone()));
        }
        if let Some(ref path) = module.bytecode_compressed {
            files.push((WorkProductFileKind::BytecodeCompressed, path.clone()));
        }

        save_trans_partition(sess, dep_graph, &module.name, &files);
    }
}

fn produce_final_output_artifacts(sess: &Session,
                                  compiled_modules: &CompiledModules,
                                  crate_output: &OutputFilenames) {
    let mut user_wants_bitcode = false;
    let mut user_wants_objects = false;

    // Produce final compile outputs.
    let copy_gracefully = |from: &Path, to: &Path| {
        if let Err(e) = fs::copy(from, to) {
            sess.err(&format!("could not copy {:?} to {:?}: {}", from, to, e));
        }
    };

    let copy_if_one_unit = |output_type: OutputType,
                            keep_numbered: bool| {
        if compiled_modules.modules.len() == 1 {
            // 1) Only one codegen unit.  In this case it's no difficulty
            //    to copy `foo.0.x` to `foo.x`.
            let module_name = Some(&compiled_modules.modules[0].name[..]);
            let path = crate_output.temp_path(output_type, module_name);
            copy_gracefully(&path,
                            &crate_output.path(output_type));
            if !sess.opts.cg.save_temps && !keep_numbered {
                // The user just wants `foo.x`, not `foo.#module-name#.x`.
                remove(sess, &path);
            }
        } else {
            let ext = crate_output.temp_path(output_type, None)
                                  .extension()
                                  .unwrap()
                                  .to_str()
                                  .unwrap()
                                  .to_owned();

            if crate_output.outputs.contains_key(&output_type) {
                // 2) Multiple codegen units, with `--emit foo=some_name`.  We have
                //    no good solution for this case, so warn the user.
                sess.warn(&format!("ignoring emit path because multiple .{} files \
                                    were produced", ext));
            } else if crate_output.single_output_file.is_some() {
                // 3) Multiple codegen units, with `-o some_name`.  We have
                //    no good solution for this case, so warn the user.
                sess.warn(&format!("ignoring -o because multiple .{} files \
                                    were produced", ext));
            } else {
                // 4) Multiple codegen units, but no explicit name.  We
                //    just leave the `foo.0.x` files in place.
                // (We don't have to do any work in this case.)
            }
        }
    };

    // Flag to indicate whether the user explicitly requested bitcode.
    // Otherwise, we produced it only as a temporary output, and will need
    // to get rid of it.
    for output_type in crate_output.outputs.keys() {
        match *output_type {
            OutputType::Bitcode => {
                user_wants_bitcode = true;
                // Copy to .bc, but always keep the .0.bc.  There is a later
                // check to figure out if we should delete .0.bc files, or keep
                // them for making an rlib.
                copy_if_one_unit(OutputType::Bitcode, true);
            }
            OutputType::LlvmAssembly => {
                copy_if_one_unit(OutputType::LlvmAssembly, false);
            }
            OutputType::Assembly => {
                copy_if_one_unit(OutputType::Assembly, false);
            }
            OutputType::Object => {
                user_wants_objects = true;
                copy_if_one_unit(OutputType::Object, true);
            }
            OutputType::Mir |
            OutputType::Metadata |
            OutputType::Exe |
            OutputType::DepInfo => {}
        }
    }

    // Clean up unwanted temporary files.

    // We create the following files by default:
    //  - #crate#.#module-name#.bc
    //  - #crate#.#module-name#.o
    //  - #crate#.crate.metadata.bc
    //  - #crate#.crate.metadata.o
    //  - #crate#.o (linked from crate.##.o)
    //  - #crate#.bc (copied from crate.##.bc)
    // We may create additional files if requested by the user (through
    // `-C save-temps` or `--emit=` flags).

    if !sess.opts.cg.save_temps {
        // Remove the temporary .#module-name#.o objects.  If the user didn't
        // explicitly request bitcode (with --emit=bc), and the bitcode is not
        // needed for building an rlib, then we must remove .#module-name#.bc as
        // well.

        // Specific rules for keeping .#module-name#.bc:
        //  - If the user requested bitcode (`user_wants_bitcode`), and
        //    codegen_units > 1, then keep it.
        //  - If the user requested bitcode but codegen_units == 1, then we
        //    can toss .#module-name#.bc because we copied it to .bc earlier.
        //  - If we're not building an rlib and the user didn't request
        //    bitcode, then delete .#module-name#.bc.
        // If you change how this works, also update back::link::link_rlib,
        // where .#module-name#.bc files are (maybe) deleted after making an
        // rlib.
        let needs_crate_object = crate_output.outputs.contains_key(&OutputType::Exe);

        let keep_numbered_bitcode = user_wants_bitcode && sess.codegen_units() > 1;

        let keep_numbered_objects = needs_crate_object ||
                (user_wants_objects && sess.codegen_units() > 1);

        for module in compiled_modules.modules.iter() {
            if let Some(ref path) = module.object {
                if !keep_numbered_objects {
                    remove(sess, path);
                }
            }

            if let Some(ref path) = module.bytecode {
                if !keep_numbered_bitcode {
                    remove(sess, path);
                }
            }
        }

        if !user_wants_bitcode {
            if let Some(ref path) = compiled_modules.metadata_module.bytecode {
                remove(sess, &path);
            }

            if let Some(ref allocator_module) = compiled_modules.allocator_module {
                if let Some(ref path) = allocator_module.bytecode {
                    remove(sess, path);
                }
            }
        }
    }

    // We leave the following files around by default:
    //  - #crate#.o
    //  - #crate#.crate.metadata.o
    //  - #crate#.bc
    // These are used in linking steps and will be cleaned up afterward.
}

pub fn dump_incremental_data(trans: &CrateTranslation) {
    println!("[incremental] Re-using {} out of {} modules",
              trans.modules.iter().filter(|m| m.pre_existing).count(),
              trans.modules.len());
}

enum WorkItem {
    Optimize(ModuleTranslation),
    LTO(lto::LtoModuleTranslation),
}

impl WorkItem {
    fn kind(&self) -> ModuleKind {
        match *self {
            WorkItem::Optimize(ref m) => m.kind,
            WorkItem::LTO(_) => ModuleKind::Regular,
        }
    }

    fn name(&self) -> String {
        match *self {
            WorkItem::Optimize(ref m) => format!("optimize: {}", m.name),
            WorkItem::LTO(ref m) => format!("lto: {}", m.name()),
        }
    }
}

enum WorkItemResult {
    Compiled(CompiledModule),
    NeedsLTO(ModuleTranslation),
}

fn execute_work_item(cgcx: &CodegenContext,
                     work_item: WorkItem,
                     timeline: &mut Timeline)
    -> Result<WorkItemResult, FatalError>
{
    let diag_handler = cgcx.create_diag_handler();
    let config = cgcx.config(work_item.kind());
    let mtrans = match work_item {
        WorkItem::Optimize(mtrans) => mtrans,
        WorkItem::LTO(mut lto) => {
            unsafe {
                let module = lto.optimize(cgcx, timeline)?;
                let module = codegen(cgcx, &diag_handler, module, config, timeline)?;
                return Ok(WorkItemResult::Compiled(module))
            }
        }
    };
    let module_name = mtrans.name.clone();

    let pre_existing = match mtrans.source {
        ModuleSource::Translated(_) => None,
        ModuleSource::Preexisting(ref wp) => Some(wp.clone()),
    };

    if let Some(wp) = pre_existing {
        let incr_comp_session_dir = cgcx.incr_comp_session_dir
                                        .as_ref()
                                        .unwrap();
        let name = &mtrans.name;
        let mut object = None;
        let mut bytecode = None;
        let mut bytecode_compressed = None;
        for (kind, saved_file) in wp.saved_files {
            let obj_out = match kind {
                WorkProductFileKind::Object => {
                    let path = cgcx.output_filenames.temp_path(OutputType::Object, Some(name));
                    object = Some(path.clone());
                    path
                }
                WorkProductFileKind::Bytecode => {
                    let path = cgcx.output_filenames.temp_path(OutputType::Bitcode, Some(name));
                    bytecode = Some(path.clone());
                    path
                }
                WorkProductFileKind::BytecodeCompressed => {
                    let path = cgcx.output_filenames.temp_path(OutputType::Bitcode, Some(name))
                        .with_extension(RLIB_BYTECODE_EXTENSION);
                    bytecode_compressed = Some(path.clone());
                    path
                }
            };
            let source_file = in_incr_comp_dir(&incr_comp_session_dir,
                                               &saved_file);
            debug!("copying pre-existing module `{}` from {:?} to {}",
                   mtrans.name,
                   source_file,
                   obj_out.display());
            match link_or_copy(&source_file, &obj_out) {
                Ok(_) => { }
                Err(err) => {
                    diag_handler.err(&format!("unable to copy {} to {}: {}",
                                              source_file.display(),
                                              obj_out.display(),
                                              err));
                }
            }
        }
        assert_eq!(object.is_some(), config.emit_obj);
        assert_eq!(bytecode.is_some(), config.emit_bc);
        assert_eq!(bytecode_compressed.is_some(), config.emit_bc_compressed);

        Ok(WorkItemResult::Compiled(CompiledModule {
            llmod_id: mtrans.llmod_id.clone(),
            name: module_name,
            kind: ModuleKind::Regular,
            pre_existing: true,
            object,
            bytecode,
            bytecode_compressed,
        }))
    } else {
        debug!("llvm-optimizing {:?}", module_name);

        unsafe {
            optimize(cgcx, &diag_handler, &mtrans, config, timeline)?;

            let lto = cgcx.lto;

            let auto_thin_lto =
                cgcx.thinlto &&
                cgcx.total_cgus > 1 &&
                mtrans.kind != ModuleKind::Allocator;

            // If we're a metadata module we never participate in LTO.
            //
            // If LTO was explicitly requested on the command line, we always
            // LTO everything else.
            //
            // If LTO *wasn't* explicitly requested and we're not a metdata
            // module, then we may automatically do ThinLTO if we've got
            // multiple codegen units. Note, however, that the allocator module
            // doesn't participate here automatically because of linker
            // shenanigans later on.
            if mtrans.kind == ModuleKind::Metadata || (!lto && !auto_thin_lto) {
                let module = codegen(cgcx, &diag_handler, mtrans, config, timeline)?;
                Ok(WorkItemResult::Compiled(module))
            } else {
                Ok(WorkItemResult::NeedsLTO(mtrans))
            }
        }
    }
}

enum Message {
    Token(io::Result<Acquired>),
    NeedsLTO {
        result: ModuleTranslation,
        worker_id: usize,
    },
    Done {
        result: Result<CompiledModule, ()>,
        worker_id: usize,
    },
    TranslationDone {
        llvm_work_item: WorkItem,
        cost: u64,
    },
    TranslationComplete,
    TranslateItem,
}

struct Diagnostic {
    msg: String,
    code: Option<DiagnosticId>,
    lvl: Level,
}

#[derive(PartialEq, Clone, Copy, Debug)]
enum MainThreadWorkerState {
    Idle,
    Translating,
    LLVMing,
}

fn start_executing_work(tcx: TyCtxt,
                        crate_info: &CrateInfo,
                        shared_emitter: SharedEmitter,
                        trans_worker_send: Sender<Message>,
                        coordinator_receive: Receiver<Box<Any + Send>>,
                        total_cgus: usize,
                        jobserver: Client,
                        time_graph: Option<TimeGraph>,
                        modules_config: Arc<ModuleConfig>,
                        metadata_config: Arc<ModuleConfig>,
                        allocator_config: Arc<ModuleConfig>)
                        -> thread::JoinHandle<Result<CompiledModules, ()>> {
    let coordinator_send = tcx.tx_to_llvm_workers.clone();
    let mut exported_symbols = FxHashMap();
    exported_symbols.insert(LOCAL_CRATE, tcx.exported_symbols(LOCAL_CRATE));
    for &cnum in tcx.crates().iter() {
        exported_symbols.insert(cnum, tcx.exported_symbols(cnum));
    }
    let exported_symbols = Arc::new(exported_symbols);
    let sess = tcx.sess;

    // First up, convert our jobserver into a helper thread so we can use normal
    // mpsc channels to manage our messages and such.
    // After we've requested tokens then we'll, when we can,
    // get tokens on `coordinator_receive` which will
    // get managed in the main loop below.
    let coordinator_send2 = coordinator_send.clone();
    let helper = jobserver.into_helper_thread(move |token| {
        drop(coordinator_send2.send(Box::new(Message::Token(token))));
    }).expect("failed to spawn helper thread");

    let mut each_linked_rlib_for_lto = Vec::new();
    drop(link::each_linked_rlib(sess, crate_info, &mut |cnum, path| {
        if link::ignored_for_lto(sess, crate_info, cnum) {
            return
        }
        each_linked_rlib_for_lto.push((cnum, path.to_path_buf()));
    }));

    let crate_types = sess.crate_types.borrow();
    let only_rlib = crate_types.len() == 1 &&
        crate_types[0] == config::CrateTypeRlib;

    let wasm_import_memory =
        attr::contains_name(&tcx.hir.krate().attrs, "wasm_import_memory");

    let cgcx = CodegenContext {
        crate_types: sess.crate_types.borrow().clone(),
        each_linked_rlib_for_lto,
        // If we're only building an rlibc then allow the LTO flag to be passed
        // but don't actually do anything, the full LTO will happen later
        lto: sess.lto() && !only_rlib,

        // Enable ThinLTO if requested, but only if the target we're compiling
        // for doesn't require full LTO. Some targets require one LLVM module
        // (they effectively don't have a linker) so it's up to us to use LTO to
        // link everything together.
        thinlto: sess.thinlto() &&
            !sess.target.target.options.requires_lto &&
            unsafe { llvm::LLVMRustThinLTOAvailable() },

        no_landing_pads: sess.no_landing_pads(),
        fewer_names: sess.fewer_names(),
        save_temps: sess.opts.cg.save_temps,
        opts: Arc::new(sess.opts.clone()),
        time_passes: sess.time_passes(),
        exported_symbols,
        plugin_passes: sess.plugin_llvm_passes.borrow().clone(),
        remark: sess.opts.cg.remark.clone(),
        worker: 0,
        incr_comp_session_dir: sess.incr_comp_session_dir_opt().map(|r| r.clone()),
        coordinator_send,
        diag_emitter: shared_emitter.clone(),
        time_graph,
        output_filenames: tcx.output_filenames(LOCAL_CRATE),
        regular_module_config: modules_config,
        metadata_module_config: metadata_config,
        allocator_module_config: allocator_config,
        tm_factory: target_machine_factory(tcx.sess),
        total_cgus,
        msvc_imps_needed: msvc_imps_needed(tcx),
        target_pointer_width: tcx.sess.target.target.target_pointer_width.clone(),
        binaryen_linker: tcx.sess.linker_flavor() == LinkerFlavor::Binaryen,
        debuginfo: tcx.sess.opts.debuginfo,
        wasm_import_memory: wasm_import_memory,
    };

    // This is the "main loop" of parallel work happening for parallel codegen.
    // It's here that we manage parallelism, schedule work, and work with
    // messages coming from clients.
    //
    // There are a few environmental pre-conditions that shape how the system
    // is set up:
    //
    // - Error reporting only can happen on the main thread because that's the
    //   only place where we have access to the compiler `Session`.
    // - LLVM work can be done on any thread.
    // - Translation can only happen on the main thread.
    // - Each thread doing substantial work most be in possession of a `Token`
    //   from the `Jobserver`.
    // - The compiler process always holds one `Token`. Any additional `Tokens`
    //   have to be requested from the `Jobserver`.
    //
    // Error Reporting
    // ===============
    // The error reporting restriction is handled separately from the rest: We
    // set up a `SharedEmitter` the holds an open channel to the main thread.
    // When an error occurs on any thread, the shared emitter will send the
    // error message to the receiver main thread (`SharedEmitterMain`). The
    // main thread will periodically query this error message queue and emit
    // any error messages it has received. It might even abort compilation if
    // has received a fatal error. In this case we rely on all other threads
    // being torn down automatically with the main thread.
    // Since the main thread will often be busy doing translation work, error
    // reporting will be somewhat delayed, since the message queue can only be
    // checked in between to work packages.
    //
    // Work Processing Infrastructure
    // ==============================
    // The work processing infrastructure knows three major actors:
    //
    // - the coordinator thread,
    // - the main thread, and
    // - LLVM worker threads
    //
    // The coordinator thread is running a message loop. It instructs the main
    // thread about what work to do when, and it will spawn off LLVM worker
    // threads as open LLVM WorkItems become available.
    //
    // The job of the main thread is to translate CGUs into LLVM work package
    // (since the main thread is the only thread that can do this). The main
    // thread will block until it receives a message from the coordinator, upon
    // which it will translate one CGU, send it to the coordinator and block
    // again. This way the coordinator can control what the main thread is
    // doing.
    //
    // The coordinator keeps a queue of LLVM WorkItems, and when a `Token` is
    // available, it will spawn off a new LLVM worker thread and let it process
    // that a WorkItem. When a LLVM worker thread is done with its WorkItem,
    // it will just shut down, which also frees all resources associated with
    // the given LLVM module, and sends a message to the coordinator that the
    // has been completed.
    //
    // Work Scheduling
    // ===============
    // The scheduler's goal is to minimize the time it takes to complete all
    // work there is, however, we also want to keep memory consumption low
    // if possible. These two goals are at odds with each other: If memory
    // consumption were not an issue, we could just let the main thread produce
    // LLVM WorkItems at full speed, assuring maximal utilization of
    // Tokens/LLVM worker threads. However, since translation usual is faster
    // than LLVM processing, the queue of LLVM WorkItems would fill up and each
    // WorkItem potentially holds on to a substantial amount of memory.
    //
    // So the actual goal is to always produce just enough LLVM WorkItems as
    // not to starve our LLVM worker threads. That means, once we have enough
    // WorkItems in our queue, we can block the main thread, so it does not
    // produce more until we need them.
    //
    // Doing LLVM Work on the Main Thread
    // ----------------------------------
    // Since the main thread owns the compiler processes implicit `Token`, it is
    // wasteful to keep it blocked without doing any work. Therefore, what we do
    // in this case is: We spawn off an additional LLVM worker thread that helps
    // reduce the queue. The work it is doing corresponds to the implicit
    // `Token`. The coordinator will mark the main thread as being busy with
    // LLVM work. (The actual work happens on another OS thread but we just care
    // about `Tokens`, not actual threads).
    //
    // When any LLVM worker thread finishes while the main thread is marked as
    // "busy with LLVM work", we can do a little switcheroo: We give the Token
    // of the just finished thread to the LLVM worker thread that is working on
    // behalf of the main thread's implicit Token, thus freeing up the main
    // thread again. The coordinator can then again decide what the main thread
    // should do. This allows the coordinator to make decisions at more points
    // in time.
    //
    // Striking a Balance between Throughput and Memory Consumption
    // ------------------------------------------------------------
    // Since our two goals, (1) use as many Tokens as possible and (2) keep
    // memory consumption as low as possible, are in conflict with each other,
    // we have to find a trade off between them. Right now, the goal is to keep
    // all workers busy, which means that no worker should find the queue empty
    // when it is ready to start.
    // How do we do achieve this? Good question :) We actually never know how
    // many `Tokens` are potentially available so it's hard to say how much to
    // fill up the queue before switching the main thread to LLVM work. Also we
    // currently don't have a means to estimate how long a running LLVM worker
    // will still be busy with it's current WorkItem. However, we know the
    // maximal count of available Tokens that makes sense (=the number of CPU
    // cores), so we can take a conservative guess. The heuristic we use here
    // is implemented in the `queue_full_enough()` function.
    //
    // Some Background on Jobservers
    // -----------------------------
    // It's worth also touching on the management of parallelism here. We don't
    // want to just spawn a thread per work item because while that's optimal
    // parallelism it may overload a system with too many threads or violate our
    // configuration for the maximum amount of cpu to use for this process. To
    // manage this we use the `jobserver` crate.
    //
    // Job servers are an artifact of GNU make and are used to manage
    // parallelism between processes. A jobserver is a glorified IPC semaphore
    // basically. Whenever we want to run some work we acquire the semaphore,
    // and whenever we're done with that work we release the semaphore. In this
    // manner we can ensure that the maximum number of parallel workers is
    // capped at any one point in time.
    //
    // LTO and the coordinator thread
    // ------------------------------
    //
    // The final job the coordinator thread is responsible for is managing LTO
    // and how that works. When LTO is requested what we'll to is collect all
    // optimized LLVM modules into a local vector on the coordinator. Once all
    // modules have been translated and optimized we hand this to the `lto`
    // module for further optimization. The `lto` module will return back a list
    // of more modules to work on, which the coordinator will continue to spawn
    // work for.
    //
    // Each LLVM module is automatically sent back to the coordinator for LTO if
    // necessary. There's already optimizations in place to avoid sending work
    // back to the coordinator if LTO isn't requested.
    return thread::spawn(move || {
        // We pretend to be within the top-level LLVM time-passes task here:
        set_time_depth(1);

        let max_workers = ::num_cpus::get();
        let mut worker_id_counter = 0;
        let mut free_worker_ids = Vec::new();
        let mut get_worker_id = |free_worker_ids: &mut Vec<usize>| {
            if let Some(id) = free_worker_ids.pop() {
                id
            } else {
                let id = worker_id_counter;
                worker_id_counter += 1;
                id
            }
        };

        // This is where we collect codegen units that have gone all the way
        // through translation and LLVM.
        let mut compiled_modules = vec![];
        let mut compiled_metadata_module = None;
        let mut compiled_allocator_module = None;
        let mut needs_lto = Vec::new();
        let mut started_lto = false;

        // This flag tracks whether all items have gone through translations
        let mut translation_done = false;

        // This is the queue of LLVM work items that still need processing.
        let mut work_items = Vec::<(WorkItem, u64)>::new();

        // This are the Jobserver Tokens we currently hold. Does not include
        // the implicit Token the compiler process owns no matter what.
        let mut tokens = Vec::new();

        let mut main_thread_worker_state = MainThreadWorkerState::Idle;
        let mut running = 0;

        let mut llvm_start_time = None;

        // Run the message loop while there's still anything that needs message
        // processing:
        while !translation_done ||
              work_items.len() > 0 ||
              running > 0 ||
              needs_lto.len() > 0 ||
              main_thread_worker_state != MainThreadWorkerState::Idle {

            // While there are still CGUs to be translated, the coordinator has
            // to decide how to utilize the compiler processes implicit Token:
            // For translating more CGU or for running them through LLVM.
            if !translation_done {
                if main_thread_worker_state == MainThreadWorkerState::Idle {
                    if !queue_full_enough(work_items.len(), running, max_workers) {
                        // The queue is not full enough, translate more items:
                        if let Err(_) = trans_worker_send.send(Message::TranslateItem) {
                            panic!("Could not send Message::TranslateItem to main thread")
                        }
                        main_thread_worker_state = MainThreadWorkerState::Translating;
                    } else {
                        // The queue is full enough to not let the worker
                        // threads starve. Use the implicit Token to do some
                        // LLVM work too.
                        let (item, _) = work_items.pop()
                            .expect("queue empty - queue_full_enough() broken?");
                        let cgcx = CodegenContext {
                            worker: get_worker_id(&mut free_worker_ids),
                            .. cgcx.clone()
                        };
                        maybe_start_llvm_timer(cgcx.config(item.kind()),
                                               &mut llvm_start_time);
                        main_thread_worker_state = MainThreadWorkerState::LLVMing;
                        spawn_work(cgcx, item);
                    }
                }
            } else {
                // If we've finished everything related to normal translation
                // then it must be the case that we've got some LTO work to do.
                // Perform the serial work here of figuring out what we're
                // going to LTO and then push a bunch of work items onto our
                // queue to do LTO
                if work_items.len() == 0 &&
                   running == 0 &&
                   main_thread_worker_state == MainThreadWorkerState::Idle {
                    assert!(!started_lto);
                    assert!(needs_lto.len() > 0);
                    started_lto = true;
                    let modules = mem::replace(&mut needs_lto, Vec::new());
                    for (work, cost) in generate_lto_work(&cgcx, modules) {
                        let insertion_index = work_items
                            .binary_search_by_key(&cost, |&(_, cost)| cost)
                            .unwrap_or_else(|e| e);
                        work_items.insert(insertion_index, (work, cost));
                        helper.request_token();
                    }
                }

                // In this branch, we know that everything has been translated,
                // so it's just a matter of determining whether the implicit
                // Token is free to use for LLVM work.
                match main_thread_worker_state {
                    MainThreadWorkerState::Idle => {
                        if let Some((item, _)) = work_items.pop() {
                            let cgcx = CodegenContext {
                                worker: get_worker_id(&mut free_worker_ids),
                                .. cgcx.clone()
                            };
                            maybe_start_llvm_timer(cgcx.config(item.kind()),
                                                   &mut llvm_start_time);
                            main_thread_worker_state = MainThreadWorkerState::LLVMing;
                            spawn_work(cgcx, item);
                        } else {
                            // There is no unstarted work, so let the main thread
                            // take over for a running worker. Otherwise the
                            // implicit token would just go to waste.
                            // We reduce the `running` counter by one. The
                            // `tokens.truncate()` below will take care of
                            // giving the Token back.
                            debug_assert!(running > 0);
                            running -= 1;
                            main_thread_worker_state = MainThreadWorkerState::LLVMing;
                        }
                    }
                    MainThreadWorkerState::Translating => {
                        bug!("trans worker should not be translating after \
                              translation was already completed")
                    }
                    MainThreadWorkerState::LLVMing => {
                        // Already making good use of that token
                    }
                }
            }

            // Spin up what work we can, only doing this while we've got available
            // parallelism slots and work left to spawn.
            while work_items.len() > 0 && running < tokens.len() {
                let (item, _) = work_items.pop().unwrap();

                maybe_start_llvm_timer(cgcx.config(item.kind()),
                                       &mut llvm_start_time);

                let cgcx = CodegenContext {
                    worker: get_worker_id(&mut free_worker_ids),
                    .. cgcx.clone()
                };

                spawn_work(cgcx, item);
                running += 1;
            }

            // Relinquish accidentally acquired extra tokens
            tokens.truncate(running);

            let msg = coordinator_receive.recv().unwrap();
            match *msg.downcast::<Message>().ok().unwrap() {
                // Save the token locally and the next turn of the loop will use
                // this to spawn a new unit of work, or it may get dropped
                // immediately if we have no more work to spawn.
                Message::Token(token) => {
                    match token {
                        Ok(token) => {
                            tokens.push(token);

                            if main_thread_worker_state == MainThreadWorkerState::LLVMing {
                                // If the main thread token is used for LLVM work
                                // at the moment, we turn that thread into a regular
                                // LLVM worker thread, so the main thread is free
                                // to react to translation demand.
                                main_thread_worker_state = MainThreadWorkerState::Idle;
                                running += 1;
                            }
                        }
                        Err(e) => {
                            let msg = &format!("failed to acquire jobserver token: {}", e);
                            shared_emitter.fatal(msg);
                            // Exit the coordinator thread
                            panic!("{}", msg)
                        }
                    }
                }

                Message::TranslationDone { llvm_work_item, cost } => {
                    // We keep the queue sorted by estimated processing cost,
                    // so that more expensive items are processed earlier. This
                    // is good for throughput as it gives the main thread more
                    // time to fill up the queue and it avoids scheduling
                    // expensive items to the end.
                    // Note, however, that this is not ideal for memory
                    // consumption, as LLVM module sizes are not evenly
                    // distributed.
                    let insertion_index =
                        work_items.binary_search_by_key(&cost, |&(_, cost)| cost);
                    let insertion_index = match insertion_index {
                        Ok(idx) | Err(idx) => idx
                    };
                    work_items.insert(insertion_index, (llvm_work_item, cost));

                    helper.request_token();
                    assert_eq!(main_thread_worker_state,
                               MainThreadWorkerState::Translating);
                    main_thread_worker_state = MainThreadWorkerState::Idle;
                }

                Message::TranslationComplete => {
                    translation_done = true;
                    assert_eq!(main_thread_worker_state,
                               MainThreadWorkerState::Translating);
                    main_thread_worker_state = MainThreadWorkerState::Idle;
                }

                // If a thread exits successfully then we drop a token associated
                // with that worker and update our `running` count. We may later
                // re-acquire a token to continue running more work. We may also not
                // actually drop a token here if the worker was running with an
                // "ephemeral token"
                //
                // Note that if the thread failed that means it panicked, so we
                // abort immediately.
                Message::Done { result: Ok(compiled_module), worker_id } => {
                    if main_thread_worker_state == MainThreadWorkerState::LLVMing {
                        main_thread_worker_state = MainThreadWorkerState::Idle;
                    } else {
                        running -= 1;
                    }

                    free_worker_ids.push(worker_id);

                    match compiled_module.kind {
                        ModuleKind::Regular => {
                            compiled_modules.push(compiled_module);
                        }
                        ModuleKind::Metadata => {
                            assert!(compiled_metadata_module.is_none());
                            compiled_metadata_module = Some(compiled_module);
                        }
                        ModuleKind::Allocator => {
                            assert!(compiled_allocator_module.is_none());
                            compiled_allocator_module = Some(compiled_module);
                        }
                    }
                }
                Message::NeedsLTO { result, worker_id } => {
                    assert!(!started_lto);
                    if main_thread_worker_state == MainThreadWorkerState::LLVMing {
                        main_thread_worker_state = MainThreadWorkerState::Idle;
                    } else {
                        running -= 1;
                    }

                    free_worker_ids.push(worker_id);
                    needs_lto.push(result);
                }
                Message::Done { result: Err(()), worker_id: _ } => {
                    shared_emitter.fatal("aborting due to worker thread failure");
                    // Exit the coordinator thread
                    return Err(())
                }
                Message::TranslateItem => {
                    bug!("the coordinator should not receive translation requests")
                }
            }
        }

        if let Some(llvm_start_time) = llvm_start_time {
            let total_llvm_time = Instant::now().duration_since(llvm_start_time);
            // This is the top-level timing for all of LLVM, set the time-depth
            // to zero.
            set_time_depth(0);
            print_time_passes_entry(cgcx.time_passes,
                                    "LLVM passes",
                                    total_llvm_time);
        }

        // Regardless of what order these modules completed in, report them to
        // the backend in the same order every time to ensure that we're handing
        // out deterministic results.
        compiled_modules.sort_by(|a, b| a.name.cmp(&b.name));

        let compiled_metadata_module = compiled_metadata_module
            .expect("Metadata module not compiled?");

        Ok(CompiledModules {
            modules: compiled_modules,
            metadata_module: compiled_metadata_module,
            allocator_module: compiled_allocator_module,
        })
    });

    // A heuristic that determines if we have enough LLVM WorkItems in the
    // queue so that the main thread can do LLVM work instead of translation
    fn queue_full_enough(items_in_queue: usize,
                         workers_running: usize,
                         max_workers: usize) -> bool {
        // Tune me, plz.
        items_in_queue > 0 &&
        items_in_queue >= max_workers.saturating_sub(workers_running / 2)
    }

    fn maybe_start_llvm_timer(config: &ModuleConfig,
                              llvm_start_time: &mut Option<Instant>) {
        // We keep track of the -Ztime-passes output manually,
        // since the closure-based interface does not fit well here.
        if config.time_passes {
            if llvm_start_time.is_none() {
                *llvm_start_time = Some(Instant::now());
            }
        }
    }
}

pub const TRANS_WORKER_ID: usize = ::std::usize::MAX;
pub const TRANS_WORKER_TIMELINE: time_graph::TimelineId =
    time_graph::TimelineId(TRANS_WORKER_ID);
pub const TRANS_WORK_PACKAGE_KIND: time_graph::WorkPackageKind =
    time_graph::WorkPackageKind(&["#DE9597", "#FED1D3", "#FDC5C7", "#B46668", "#88494B"]);
const LLVM_WORK_PACKAGE_KIND: time_graph::WorkPackageKind =
    time_graph::WorkPackageKind(&["#7DB67A", "#C6EEC4", "#ACDAAA", "#579354", "#3E6F3C"]);

fn spawn_work(cgcx: CodegenContext, work: WorkItem) {
    let depth = time_depth();

    thread::spawn(move || {
        set_time_depth(depth);

        // Set up a destructor which will fire off a message that we're done as
        // we exit.
        struct Bomb {
            coordinator_send: Sender<Box<Any + Send>>,
            result: Option<WorkItemResult>,
            worker_id: usize,
        }
        impl Drop for Bomb {
            fn drop(&mut self) {
                let worker_id = self.worker_id;
                let msg = match self.result.take() {
                    Some(WorkItemResult::Compiled(m)) => {
                        Message::Done { result: Ok(m), worker_id }
                    }
                    Some(WorkItemResult::NeedsLTO(m)) => {
                        Message::NeedsLTO { result: m, worker_id }
                    }
                    None => Message::Done { result: Err(()), worker_id }
                };
                drop(self.coordinator_send.send(Box::new(msg)));
            }
        }

        let mut bomb = Bomb {
            coordinator_send: cgcx.coordinator_send.clone(),
            result: None,
            worker_id: cgcx.worker,
        };

        // Execute the work itself, and if it finishes successfully then flag
        // ourselves as a success as well.
        //
        // Note that we ignore any `FatalError` coming out of `execute_work_item`,
        // as a diagnostic was already sent off to the main thread - just
        // surface that there was an error in this worker.
        bomb.result = {
            let timeline = cgcx.time_graph.as_ref().map(|tg| {
                tg.start(time_graph::TimelineId(cgcx.worker),
                         LLVM_WORK_PACKAGE_KIND,
                         &work.name())
            });
            let mut timeline = timeline.unwrap_or(Timeline::noop());
            execute_work_item(&cgcx, work, &mut timeline).ok()
        };
    });
}

pub fn run_assembler(sess: &Session, outputs: &OutputFilenames) {
    let (pname, mut cmd, _) = get_linker(sess);

    for arg in &sess.target.target.options.asm_args {
        cmd.arg(arg);
    }

    cmd.arg("-c").arg("-o").arg(&outputs.path(OutputType::Object))
                           .arg(&outputs.temp_path(OutputType::Assembly, None));
    debug!("{:?}", cmd);

    match cmd.output() {
        Ok(prog) => {
            if !prog.status.success() {
                let mut note = prog.stderr.clone();
                note.extend_from_slice(&prog.stdout);

                sess.struct_err(&format!("linking with `{}` failed: {}",
                                         pname.display(),
                                         prog.status))
                    .note(&format!("{:?}", &cmd))
                    .note(str::from_utf8(&note[..]).unwrap())
                    .emit();
                sess.abort_if_errors();
            }
        },
        Err(e) => {
            sess.err(&format!("could not exec the linker `{}`: {}", pname.display(), e));
            sess.abort_if_errors();
        }
    }
}

pub unsafe fn with_llvm_pmb(llmod: ModuleRef,
                            config: &ModuleConfig,
                            opt_level: llvm::CodeGenOptLevel,
                            f: &mut FnMut(llvm::PassManagerBuilderRef)) {
    // Create the PassManagerBuilder for LLVM. We configure it with
    // reasonable defaults and prepare it to actually populate the pass
    // manager.
    let builder = llvm::LLVMPassManagerBuilderCreate();
    let opt_size = config.opt_size.unwrap_or(llvm::CodeGenOptSizeNone);
    let inline_threshold = config.inline_threshold;

    llvm::LLVMRustConfigurePassManagerBuilder(builder,
                                              opt_level,
                                              config.merge_functions,
                                              config.vectorize_slp,
                                              config.vectorize_loop);
    llvm::LLVMPassManagerBuilderSetSizeLevel(builder, opt_size as u32);

    if opt_size != llvm::CodeGenOptSizeNone {
        llvm::LLVMPassManagerBuilderSetDisableUnrollLoops(builder, 1);
    }

    llvm::LLVMRustAddBuilderLibraryInfo(builder, llmod, config.no_builtins);

    // Here we match what clang does (kinda). For O0 we only inline
    // always-inline functions (but don't add lifetime intrinsics), at O1 we
    // inline with lifetime intrinsics, and O2+ we add an inliner with a
    // thresholds copied from clang.
    match (opt_level, opt_size, inline_threshold) {
        (.., Some(t)) => {
            llvm::LLVMPassManagerBuilderUseInlinerWithThreshold(builder, t as u32);
        }
        (llvm::CodeGenOptLevel::Aggressive, ..) => {
            llvm::LLVMPassManagerBuilderUseInlinerWithThreshold(builder, 275);
        }
        (_, llvm::CodeGenOptSizeDefault, _) => {
            llvm::LLVMPassManagerBuilderUseInlinerWithThreshold(builder, 75);
        }
        (_, llvm::CodeGenOptSizeAggressive, _) => {
            llvm::LLVMPassManagerBuilderUseInlinerWithThreshold(builder, 25);
        }
        (llvm::CodeGenOptLevel::None, ..) => {
            llvm::LLVMRustAddAlwaysInlinePass(builder, false);
        }
        (llvm::CodeGenOptLevel::Less, ..) => {
            llvm::LLVMRustAddAlwaysInlinePass(builder, true);
        }
        (llvm::CodeGenOptLevel::Default, ..) => {
            llvm::LLVMPassManagerBuilderUseInlinerWithThreshold(builder, 225);
        }
        (llvm::CodeGenOptLevel::Other, ..) => {
            bug!("CodeGenOptLevel::Other selected")
        }
    }

    f(builder);
    llvm::LLVMPassManagerBuilderDispose(builder);
}


enum SharedEmitterMessage {
    Diagnostic(Diagnostic),
    InlineAsmError(u32, String),
    AbortIfErrors,
    Fatal(String),
}

#[derive(Clone)]
pub struct SharedEmitter {
    sender: Sender<SharedEmitterMessage>,
}

pub struct SharedEmitterMain {
    receiver: Receiver<SharedEmitterMessage>,
}

impl SharedEmitter {
    pub fn new() -> (SharedEmitter, SharedEmitterMain) {
        let (sender, receiver) = channel();

        (SharedEmitter { sender }, SharedEmitterMain { receiver })
    }

    fn inline_asm_error(&self, cookie: u32, msg: String) {
        drop(self.sender.send(SharedEmitterMessage::InlineAsmError(cookie, msg)));
    }

    fn fatal(&self, msg: &str) {
        drop(self.sender.send(SharedEmitterMessage::Fatal(msg.to_string())));
    }
}

impl Emitter for SharedEmitter {
    fn emit(&mut self, db: &DiagnosticBuilder) {
        drop(self.sender.send(SharedEmitterMessage::Diagnostic(Diagnostic {
            msg: db.message(),
            code: db.code.clone(),
            lvl: db.level,
        })));
        for child in &db.children {
            drop(self.sender.send(SharedEmitterMessage::Diagnostic(Diagnostic {
                msg: child.message(),
                code: None,
                lvl: child.level,
            })));
        }
        drop(self.sender.send(SharedEmitterMessage::AbortIfErrors));
    }
}

impl SharedEmitterMain {
    pub fn check(&self, sess: &Session, blocking: bool) {
        loop {
            let message = if blocking {
                match self.receiver.recv() {
                    Ok(message) => Ok(message),
                    Err(_) => Err(()),
                }
            } else {
                match self.receiver.try_recv() {
                    Ok(message) => Ok(message),
                    Err(_) => Err(()),
                }
            };

            match message {
                Ok(SharedEmitterMessage::Diagnostic(diag)) => {
                    let handler = sess.diagnostic();
                    match diag.code {
                        Some(ref code) => {
                            handler.emit_with_code(&MultiSpan::new(),
                                                   &diag.msg,
                                                   code.clone(),
                                                   diag.lvl);
                        }
                        None => {
                            handler.emit(&MultiSpan::new(),
                                         &diag.msg,
                                         diag.lvl);
                        }
                    }
                }
                Ok(SharedEmitterMessage::InlineAsmError(cookie, msg)) => {
                    match Mark::from_u32(cookie).expn_info() {
                        Some(ei) => sess.span_err(ei.call_site, &msg),
                        None     => sess.err(&msg),
                    }
                }
                Ok(SharedEmitterMessage::AbortIfErrors) => {
                    sess.abort_if_errors();
                }
                Ok(SharedEmitterMessage::Fatal(msg)) => {
                    sess.fatal(&msg);
                }
                Err(_) => {
                    break;
                }
            }

        }
    }
}

pub struct OngoingCrateTranslation {
    crate_name: Symbol,
    link: LinkMeta,
    metadata: EncodedMetadata,
    windows_subsystem: Option<String>,
    linker_info: LinkerInfo,
    no_integrated_as: bool,
    crate_info: CrateInfo,
    time_graph: Option<TimeGraph>,
    coordinator_send: Sender<Box<Any + Send>>,
    trans_worker_receive: Receiver<Message>,
    shared_emitter_main: SharedEmitterMain,
    future: thread::JoinHandle<Result<CompiledModules, ()>>,
    output_filenames: Arc<OutputFilenames>,
}

impl OngoingCrateTranslation {
    pub fn join(self, sess: &Session, dep_graph: &DepGraph) -> CrateTranslation {
        self.shared_emitter_main.check(sess, true);
        let compiled_modules = match self.future.join() {
            Ok(Ok(compiled_modules)) => compiled_modules,
            Ok(Err(())) => {
                sess.abort_if_errors();
                panic!("expected abort due to worker thread errors")
            },
            Err(_) => {
                sess.fatal("Error during translation/LLVM phase.");
            }
        };

        sess.abort_if_errors();

        if let Some(time_graph) = self.time_graph {
            time_graph.dump(&format!("{}-timings", self.crate_name));
        }

        copy_module_artifacts_into_incr_comp_cache(sess,
                                                   dep_graph,
                                                   &compiled_modules);
        produce_final_output_artifacts(sess,
                                       &compiled_modules,
                                       &self.output_filenames);

        // FIXME: time_llvm_passes support - does this use a global context or
        // something?
        if sess.codegen_units() == 1 && sess.time_llvm_passes() {
            unsafe { llvm::LLVMRustPrintPassTimings(); }
        }

        let trans = CrateTranslation {
            crate_name: self.crate_name,
            link: self.link,
            metadata: self.metadata,
            windows_subsystem: self.windows_subsystem,
            linker_info: self.linker_info,
            crate_info: self.crate_info,

            modules: compiled_modules.modules,
            allocator_module: compiled_modules.allocator_module,
            metadata_module: compiled_modules.metadata_module,
        };

        if self.no_integrated_as {
            run_assembler(sess,  &self.output_filenames);

            // HACK the linker expects the object file to be named foo.0.o but
            // `run_assembler` produces an object named just foo.o. Rename it if we
            // are going to build an executable
            if sess.opts.output_types.contains_key(&OutputType::Exe) {
                let f =  self.output_filenames.path(OutputType::Object);
                rename_or_copy_remove(&f,
                    f.with_file_name(format!("{}.0.o",
                                             f.file_stem().unwrap().to_string_lossy()))).unwrap();
            }

            // Remove assembly source, unless --save-temps was specified
            if !sess.opts.cg.save_temps {
                fs::remove_file(&self.output_filenames
                                     .temp_path(OutputType::Assembly, None)).unwrap();
            }
        }

        trans
    }

    pub fn submit_pre_translated_module_to_llvm(&self,
                                                tcx: TyCtxt,
                                                mtrans: ModuleTranslation) {
        self.wait_for_signal_to_translate_item();
        self.check_for_errors(tcx.sess);

        // These are generally cheap and won't through off scheduling.
        let cost = 0;
        submit_translated_module_to_llvm(tcx, mtrans, cost);
    }

    pub fn translation_finished(&self, tcx: TyCtxt) {
        self.wait_for_signal_to_translate_item();
        self.check_for_errors(tcx.sess);
        drop(self.coordinator_send.send(Box::new(Message::TranslationComplete)));
    }

    pub fn check_for_errors(&self, sess: &Session) {
        self.shared_emitter_main.check(sess, false);
    }

    pub fn wait_for_signal_to_translate_item(&self) {
        match self.trans_worker_receive.recv() {
            Ok(Message::TranslateItem) => {
                // Nothing to do
            }
            Ok(_) => panic!("unexpected message"),
            Err(_) => {
                // One of the LLVM threads must have panicked, fall through so
                // error handling can be reached.
            }
        }
    }
}

pub fn submit_translated_module_to_llvm(tcx: TyCtxt,
                                        mtrans: ModuleTranslation,
                                        cost: u64) {
    let llvm_work_item = WorkItem::Optimize(mtrans);
    drop(tcx.tx_to_llvm_workers.send(Box::new(Message::TranslationDone {
        llvm_work_item,
        cost,
    })));
}

fn msvc_imps_needed(tcx: TyCtxt) -> bool {
    tcx.sess.target.target.options.is_like_msvc &&
        tcx.sess.crate_types.borrow().iter().any(|ct| *ct == config::CrateTypeRlib)
}

// Create a `__imp_<symbol> = &symbol` global for every public static `symbol`.
// This is required to satisfy `dllimport` references to static data in .rlibs
// when using MSVC linker.  We do this only for data, as linker can fix up
// code references on its own.
// See #26591, #27438
fn create_msvc_imps(cgcx: &CodegenContext, llcx: ContextRef, llmod: ModuleRef) {
    if !cgcx.msvc_imps_needed {
        return
    }
    // The x86 ABI seems to require that leading underscores are added to symbol
    // names, so we need an extra underscore on 32-bit. There's also a leading
    // '\x01' here which disables LLVM's symbol mangling (e.g. no extra
    // underscores added in front).
    let prefix = if cgcx.target_pointer_width == "32" {
        "\x01__imp__"
    } else {
        "\x01__imp_"
    };
    unsafe {
        let i8p_ty = Type::i8p_llcx(llcx);
        let globals = base::iter_globals(llmod)
            .filter(|&val| {
                llvm::LLVMRustGetLinkage(val) == llvm::Linkage::ExternalLinkage &&
                    llvm::LLVMIsDeclaration(val) == 0
            })
            .map(move |val| {
                let name = CStr::from_ptr(llvm::LLVMGetValueName(val));
                let mut imp_name = prefix.as_bytes().to_vec();
                imp_name.extend(name.to_bytes());
                let imp_name = CString::new(imp_name).unwrap();
                (imp_name, val)
            })
            .collect::<Vec<_>>();
        for (imp_name, val) in globals {
            let imp = llvm::LLVMAddGlobal(llmod,
                                          i8p_ty.to_ref(),
                                          imp_name.as_ptr() as *const _);
            llvm::LLVMSetInitializer(imp, consts::ptrcast(val, i8p_ty));
            llvm::LLVMRustSetLinkage(imp, llvm::Linkage::ExternalLinkage);
        }
    }
}
