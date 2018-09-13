// Copyright 2013-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use attributes;
use back::bytecode::{self, RLIB_BYTECODE_EXTENSION};
use back::lto::{self, ModuleBuffer, ThinBuffer, SerializedModule};
use back::link::{self, get_linker, remove};
use back::command::Command;
use back::linker::LinkerInfo;
use back::symbol_export::ExportedSymbols;
use base;
use consts;
use memmap;
use rustc_incremental::{copy_cgu_workproducts_to_incr_comp_cache_dir,
                        in_incr_comp_dir, in_incr_comp_dir_sess};
use rustc::dep_graph::{WorkProduct, WorkProductId, WorkProductFileKind};
use rustc::dep_graph::cgu_reuse_tracker::CguReuseTracker;
use rustc::middle::cstore::EncodedMetadata;
use rustc::session::config::{self, OutputFilenames, OutputType, Passes, Sanitizer, Lto};
use rustc::session::Session;
use rustc::util::nodemap::FxHashMap;
use time_graph::{self, TimeGraph, Timeline};
use llvm::{self, DiagnosticInfo, PassManager, SMDiagnostic};
use llvm_util;
use {CodegenResults, ModuleCodegen, CompiledModule, ModuleKind, // ModuleLlvm,
     CachedModuleCodegen};
use CrateInfo;
use rustc::hir::def_id::{CrateNum, LOCAL_CRATE};
use rustc::ty::TyCtxt;
use rustc::util::common::{time_ext, time_depth, set_time_depth, print_time_passes_entry};
use rustc_fs_util::{path2cstr, link_or_copy};
use rustc_data_structures::small_c_str::SmallCStr;
use rustc_data_structures::svh::Svh;
use errors::{self, Handler, Level, DiagnosticBuilder, FatalError, DiagnosticId};
use errors::emitter::{Emitter};
use syntax::attr;
use syntax::ext::hygiene::Mark;
use syntax_pos::MultiSpan;
use syntax_pos::symbol::Symbol;
use type_::Type;
use context::{is_pie_binary, get_reloc_model};
use common::{C_bytes_in_context, val_ty};
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

pub const CODE_GEN_MODEL_ARGS: &[(&str, llvm::CodeModel)] = &[
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

const PRE_THIN_LTO_BC_EXT: &str = "pre-thin-lto.bc";

pub fn llvm_err(handler: &errors::Handler, msg: String) -> FatalError {
    match llvm::last_error() {
        Some(err) => handler.fatal(&format!("{}: {}", msg, err)),
        None => handler.fatal(&msg),
    }
}

pub fn write_output_file(
        handler: &errors::Handler,
        target: &'ll llvm::TargetMachine,
        pm: &llvm::PassManager<'ll>,
        m: &'ll llvm::Module,
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

pub fn create_target_machine(
    sess: &Session,
    find_features: bool,
) -> &'static mut llvm::TargetMachine {
    target_machine_factory(sess, find_features)().unwrap_or_else(|err| {
        llvm_err(sess.diagnostic(), err).raise()
    })
}

// If find_features is true this won't access `sess.crate_types` by assuming
// that `is_pie_binary` is false. When we discover LLVM target features
// `sess.crate_types` is uninitialized so we cannot access it.
pub fn target_machine_factory(sess: &Session, find_features: bool)
    -> Arc<dyn Fn() -> Result<&'static mut llvm::TargetMachine, String> + Send + Sync>
{
    let reloc_model = get_reloc_model(sess);

    let opt_level = get_llvm_opt_level(sess.opts.optimize);
    let use_softfp = sess.opts.cg.soft_float;

    let ffunction_sections = sess.target.target.options.function_sections;
    let fdata_sections = ffunction_sections;

    let code_model_arg = sess.opts.cg.code_model.as_ref().or(
        sess.target.target.options.code_model.as_ref(),
    );

    let code_model = match code_model_arg {
        Some(s) => {
            match CODE_GEN_MODEL_ARGS.iter().find(|arg| arg.0 == s) {
                Some(x) => x.1,
                _ => {
                    sess.err(&format!("{:?} is not a valid code model",
                                      code_model_arg));
                    sess.abort_if_errors();
                    bug!();
                }
            }
        }
        None => llvm::CodeModel::None,
    };

    let features = attributes::llvm_target_features(sess).collect::<Vec<_>>();
    let mut singlethread = sess.target.target.options.singlethread;

    // On the wasm target once the `atomics` feature is enabled that means that
    // we're no longer single-threaded, or otherwise we don't want LLVM to
    // lower atomic operations to single-threaded operations.
    if singlethread &&
        sess.target.target.llvm_target.contains("wasm32") &&
        features.iter().any(|s| *s == "+atomics")
    {
        singlethread = false;
    }

    let triple = SmallCStr::new(&sess.target.target.llvm_target);
    let cpu = SmallCStr::new(llvm_util::target_cpu(sess));
    let features = features.join(",");
    let features = CString::new(features).unwrap();
    let is_pie_binary = !find_features && is_pie_binary(sess);
    let trap_unreachable = sess.target.target.options.trap_unreachable;
    let emit_stack_size_section = sess.opts.debugging_opts.emit_stack_sizes;

    let asm_comments = sess.asm_comments();

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
                asm_comments,
                emit_stack_size_section,
            )
        };

        tm.ok_or_else(|| {
            format!("Could not create LLVM TargetMachine for triple: {}",
                    triple.to_str().unwrap())
        })
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

    pgo_gen: Option<String>,
    pgo_use: String,

    // Flags indicating which outputs to produce.
    pub emit_pre_thin_lto_bc: bool,
    emit_no_opt_bc: bool,
    emit_bc: bool,
    emit_bc_compressed: bool,
    emit_lto_bc: bool,
    emit_ir: bool,
    emit_asm: bool,
    emit_obj: bool,
    // Miscellaneous flags.  These are mostly copied from command-line
    // options.
    pub verify_llvm_ir: bool,
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
    no_integrated_as: bool,
    embed_bitcode: bool,
    embed_bitcode_marker: bool,
}

impl ModuleConfig {
    fn new(passes: Vec<String>) -> ModuleConfig {
        ModuleConfig {
            passes,
            opt_level: None,
            opt_size: None,

            pgo_gen: None,
            pgo_use: String::new(),

            emit_no_opt_bc: false,
            emit_pre_thin_lto_bc: false,
            emit_bc: false,
            emit_bc_compressed: false,
            emit_lto_bc: false,
            emit_ir: false,
            emit_asm: false,
            emit_obj: false,
            obj_is_bitcode: false,
            embed_bitcode: false,
            embed_bitcode_marker: false,
            no_integrated_as: false,

            verify_llvm_ir: false,
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
        self.verify_llvm_ir = sess.verify_llvm_ir();
        self.no_prepopulate_passes = sess.opts.cg.no_prepopulate_passes;
        self.no_builtins = no_builtins || sess.target.target.options.no_builtins;
        self.time_passes = sess.time_passes();
        self.inline_threshold = sess.opts.cg.inline_threshold;
        self.obj_is_bitcode = sess.target.target.options.obj_is_bitcode ||
                              sess.opts.debugging_opts.cross_lang_lto.enabled();
        let embed_bitcode = sess.target.target.options.embed_bitcode ||
                            sess.opts.debugging_opts.embed_bitcode;
        if embed_bitcode {
            match sess.opts.optimize {
                config::OptLevel::No |
                config::OptLevel::Less => {
                    self.embed_bitcode_marker = embed_bitcode;
                }
                _ => self.embed_bitcode = embed_bitcode,
            }
        }

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

/// Assembler name and command used by codegen when no_integrated_as is enabled
struct AssemblerCommand {
    name: PathBuf,
    cmd: Command,
}

/// Additional resources used by optimize_and_codegen (not module specific)
#[derive(Clone)]
pub struct CodegenContext {
    // Resources needed when running LTO
    pub time_passes: bool,
    pub lto: Lto,
    pub no_landing_pads: bool,
    pub save_temps: bool,
    pub fewer_names: bool,
    pub exported_symbols: Option<Arc<ExportedSymbols>>,
    pub opts: Arc<config::Options>,
    pub crate_types: Vec<config::CrateType>,
    pub each_linked_rlib_for_lto: Vec<(CrateNum, PathBuf)>,
    output_filenames: Arc<OutputFilenames>,
    regular_module_config: Arc<ModuleConfig>,
    metadata_module_config: Arc<ModuleConfig>,
    allocator_module_config: Arc<ModuleConfig>,
    pub tm_factory: Arc<dyn Fn() -> Result<&'static mut llvm::TargetMachine, String> + Send + Sync>,
    pub msvc_imps_needed: bool,
    pub target_pointer_width: String,
    debuginfo: config::DebugInfo,

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
    // Used to update CGU re-use information during the thinlto phase.
    pub cgu_reuse_tracker: CguReuseTracker,
    // Channel back to the main control thread to send messages to
    coordinator_send: Sender<Box<dyn Any + Send>>,
    // A reference to the TimeGraph so we can register timings. None means that
    // measuring is disabled.
    time_graph: Option<TimeGraph>,
    // The assembler command if no_integrated_as option is enabled, None otherwise
    assembler_cmd: Option<Arc<AssemblerCommand>>,
}

impl CodegenContext {
    pub fn create_diag_handler(&self) -> Handler {
        Handler::with_emitter(true, false, Box::new(self.diag_emitter.clone()))
    }

    pub(crate) fn config(&self, kind: ModuleKind) -> &ModuleConfig {
        match kind {
            ModuleKind::Regular => &self.regular_module_config,
            ModuleKind::Metadata => &self.metadata_module_config,
            ModuleKind::Allocator => &self.allocator_module_config,
        }
    }

    pub(crate) fn save_temp_bitcode(&self, module: &ModuleCodegen, name: &str) {
        if !self.save_temps {
            return
        }
        unsafe {
            let ext = format!("{}.bc", name);
            let cgu = Some(&module.name[..]);
            let path = self.output_filenames.temp_path_ext(&ext, cgu);
            let cstr = path2cstr(&path);
            let llmod = module.module_llvm.llmod();
            llvm::LLVMWriteBitcodeToFile(llmod, cstr.as_ptr());
        }
    }
}

pub struct DiagnosticHandlers<'a> {
    data: *mut (&'a CodegenContext, &'a Handler),
    llcx: &'a llvm::Context,
}

impl<'a> DiagnosticHandlers<'a> {
    pub fn new(cgcx: &'a CodegenContext,
               handler: &'a Handler,
               llcx: &'a llvm::Context) -> Self {
        let data = Box::into_raw(Box::new((cgcx, handler)));
        unsafe {
            llvm::LLVMRustSetInlineAsmDiagnosticHandler(llcx, inline_asm_handler, data as *mut _);
            llvm::LLVMContextSetDiagnosticHandler(llcx, diagnostic_handler, data as *mut _);
        }
        DiagnosticHandlers { data, llcx }
    }
}

impl<'a> Drop for DiagnosticHandlers<'a> {
    fn drop(&mut self) {
        use std::ptr::null_mut;
        unsafe {
            llvm::LLVMRustSetInlineAsmDiagnosticHandler(self.llcx, inline_asm_handler, null_mut());
            llvm::LLVMContextSetDiagnosticHandler(self.llcx, diagnostic_handler, null_mut());
            drop(Box::from_raw(self.data));
        }
    }
}

unsafe extern "C" fn report_inline_asm<'a, 'b>(cgcx: &'a CodegenContext,
                                               msg: &'b str,
                                               cookie: c_uint) {
    cgcx.diag_emitter.inline_asm_error(cookie as u32, msg.to_string());
}

unsafe extern "C" fn inline_asm_handler(diag: &SMDiagnostic,
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

unsafe extern "C" fn diagnostic_handler(info: &DiagnosticInfo, user: *mut c_void) {
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
                Passes::All => true,
                Passes::Some(ref v) => v.iter().any(|s| *s == opt.pass_name),
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
        llvm::diagnostic::PGO(diagnostic_ref) |
        llvm::diagnostic::Linker(diagnostic_ref) => {
            let msg = llvm::build_string(|s| {
                llvm::LLVMRustWriteDiagnosticInfoToString(diagnostic_ref, s)
            }).expect("non-UTF8 diagnostic");
            diag_handler.warn(&msg);
        }
        llvm::diagnostic::UnknownDiagnostic(..) => {},
    }
}

// Unsafe due to LLVM calls.
unsafe fn optimize(cgcx: &CodegenContext,
                   diag_handler: &Handler,
                   module: &ModuleCodegen,
                   config: &ModuleConfig,
                   timeline: &mut Timeline)
    -> Result<(), FatalError>
{
    let llmod = module.module_llvm.llmod();
    let llcx = &*module.module_llvm.llcx;
    let tm = &*module.module_llvm.tm;
    let _handlers = DiagnosticHandlers::new(cgcx, diag_handler, llcx);

    let module_name = module.name.clone();
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

        {
            // If we're verifying or linting, add them to the function pass
            // manager.
            let addpass = |pass_name: &str| {
                let pass_name = SmallCStr::new(pass_name);
                let pass = match llvm::LLVMRustFindAndCreatePass(pass_name.as_ptr()) {
                    Some(pass) => pass,
                    None => return false,
                };
                let pass_manager = match llvm::LLVMRustPassKind(pass) {
                    llvm::PassKind::Function => &*fpm,
                    llvm::PassKind::Module => &*mpm,
                    llvm::PassKind::Other => {
                        diag_handler.err("Encountered LLVM pass kind we can't handle");
                        return true
                    },
                };
                llvm::LLVMRustAddPass(pass_manager, pass);
                true
            };

            if config.verify_llvm_ir { assert!(addpass("verify")); }

            // Some options cause LLVM bitcode to be emitted, which uses ThinLTOBuffers, so we need
            // to make sure we run LLVM's NameAnonGlobals pass when emitting bitcode; otherwise
            // we'll get errors in LLVM.
            let using_thin_buffers = llvm::LLVMRustThinLTOAvailable() && (config.emit_bc
                || config.obj_is_bitcode || config.emit_bc_compressed || config.embed_bitcode);
            let mut have_name_anon_globals_pass = false;
            if !config.no_prepopulate_passes {
                llvm::LLVMRustAddAnalysisPasses(tm, fpm, llmod);
                llvm::LLVMRustAddAnalysisPasses(tm, mpm, llmod);
                let opt_level = config.opt_level.unwrap_or(llvm::CodeGenOptLevel::None);
                let prepare_for_thin_lto = cgcx.lto == Lto::Thin || cgcx.lto == Lto::ThinLocal ||
                    (cgcx.lto != Lto::Fat && cgcx.opts.debugging_opts.cross_lang_lto.enabled());
                have_name_anon_globals_pass = have_name_anon_globals_pass || prepare_for_thin_lto;
                if using_thin_buffers && !prepare_for_thin_lto {
                    assert!(addpass("name-anon-globals"));
                    have_name_anon_globals_pass = true;
                }
                with_llvm_pmb(llmod, &config, opt_level, prepare_for_thin_lto, &mut |b| {
                    llvm::LLVMPassManagerBuilderPopulateFunctionPassManager(b, fpm);
                    llvm::LLVMPassManagerBuilderPopulateModulePassManager(b, mpm);
                })
            }

            for pass in &config.passes {
                if !addpass(pass) {
                    diag_handler.warn(&format!("unknown pass `{}`, ignoring",
                                            pass));
                }
                if pass == "name-anon-globals" {
                    have_name_anon_globals_pass = true;
                }
            }

            for pass in &cgcx.plugin_passes {
                if !addpass(pass) {
                    diag_handler.err(&format!("a plugin asked for LLVM pass \
                                            `{}` but LLVM does not \
                                            recognize it", pass));
                }
                if pass == "name-anon-globals" {
                    have_name_anon_globals_pass = true;
                }
            }

            if using_thin_buffers && !have_name_anon_globals_pass {
                // As described above, this will probably cause an error in LLVM
                if config.no_prepopulate_passes {
                    diag_handler.err("The current compilation is going to use thin LTO buffers \
                                     without running LLVM's NameAnonGlobals pass. \
                                     This will likely cause errors in LLVM. Consider adding \
                                     -C passes=name-anon-globals to the compiler command line.");
                } else {
                    bug!("We are using thin LTO buffers without running the NameAnonGlobals pass. \
                         This will likely cause errors in LLVM and should never happen.");
                }
            }
        }

        diag_handler.abort_if_errors();

        // Finally, run the actual optimization passes
        time_ext(config.time_passes,
                 None,
                 &format!("llvm function passes [{}]", module_name.unwrap()),
                 || {
            llvm::LLVMRustRunFunctionPassManager(fpm, llmod)
        });
        timeline.record("fpm");
        time_ext(config.time_passes,
                 None,
                 &format!("llvm module passes [{}]", module_name.unwrap()),
                 || {
            llvm::LLVMRunPassManager(mpm, llmod)
        });

        // Deallocate managers that we're now done with
        llvm::LLVMDisposePassManager(fpm);
        llvm::LLVMDisposePassManager(mpm);
    }
    Ok(())
}

fn generate_lto_work(cgcx: &CodegenContext,
                     modules: Vec<ModuleCodegen>,
                     import_only_modules: Vec<(SerializedModule, WorkProduct)>)
    -> Vec<(WorkItem, u64)>
{
    let mut timeline = cgcx.time_graph.as_ref().map(|tg| {
        tg.start(CODEGEN_WORKER_TIMELINE,
                 CODEGEN_WORK_PACKAGE_KIND,
                 "generate lto")
    }).unwrap_or(Timeline::noop());
    let (lto_modules, copy_jobs) = lto::run(cgcx, modules, import_only_modules, &mut timeline)
        .unwrap_or_else(|e| e.raise());

    let lto_modules = lto_modules.into_iter().map(|module| {
        let cost = module.cost();
        (WorkItem::LTO(module), cost)
    });

    let copy_jobs = copy_jobs.into_iter().map(|wp| {
        (WorkItem::CopyPostLtoArtifacts(CachedModuleCodegen {
            name: wp.cgu_name.clone(),
            source: wp,
        }), 0)
    });

    lto_modules.chain(copy_jobs).collect()
}

unsafe fn codegen(cgcx: &CodegenContext,
                  diag_handler: &Handler,
                  module: ModuleCodegen,
                  config: &ModuleConfig,
                  timeline: &mut Timeline)
    -> Result<CompiledModule, FatalError>
{
    timeline.record("codegen");
    {
        let llmod = module.module_llvm.llmod();
        let llcx = &*module.module_llvm.llcx;
        let tm = &*module.module_llvm.tm;
        let module_name = module.name.clone();
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
        unsafe fn with_codegen<'ll, F, R>(tm: &'ll llvm::TargetMachine,
                                    llmod: &'ll llvm::Module,
                                    no_builtins: bool,
                                    f: F) -> R
            where F: FnOnce(&'ll mut PassManager<'ll>) -> R,
        {
            let cpm = llvm::LLVMCreatePassManager();
            llvm::LLVMRustAddAnalysisPasses(tm, cpm, llmod);
            llvm::LLVMRustAddLibraryInfo(cpm, llmod, no_builtins);
            f(cpm)
        }

        // If we don't have the integrated assembler, then we need to emit asm
        // from LLVM and use `gcc` to create the object file.
        let asm_to_obj = config.emit_obj && config.no_integrated_as;

        // Change what we write and cleanup based on whether obj files are
        // just llvm bitcode. In that case write bitcode, and possibly
        // delete the bitcode if it wasn't requested. Don't generate the
        // machine code, instead copy the .o file from the .bc
        let write_bc = config.emit_bc || config.obj_is_bitcode;
        let rm_bc = !config.emit_bc && config.obj_is_bitcode;
        let write_obj = config.emit_obj && !config.obj_is_bitcode && !asm_to_obj;
        let copy_bc_to_obj = config.emit_obj && config.obj_is_bitcode;

        let bc_out = cgcx.output_filenames.temp_path(OutputType::Bitcode, module_name);
        let obj_out = cgcx.output_filenames.temp_path(OutputType::Object, module_name);


        if write_bc || config.emit_bc_compressed || config.embed_bitcode {
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

            if config.embed_bitcode {
                embed_bitcode(cgcx, llcx, llmod, Some(data));
                timeline.record("embed-bc");
            }

            if config.emit_bc_compressed {
                let dst = bc_out.with_extension(RLIB_BYTECODE_EXTENSION);
                let data = bytecode::encode(&module.name, data);
                if let Err(e) = fs::write(&dst, data) {
                    diag_handler.err(&format!("failed to write bytecode: {}", e));
                }
                timeline.record("compress-bc");
            }
        } else if config.embed_bitcode_marker {
            embed_bitcode(cgcx, llcx, llmod, None);
        }

        time_ext(config.time_passes, None, &format!("codegen passes [{}]", module_name.unwrap()),
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

            if config.emit_asm || asm_to_obj {
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
                timeline.record("asm");
            }

            if write_obj {
                with_codegen(tm, llmod, config.no_builtins, |cpm| {
                    write_output_file(diag_handler, tm, cpm, llmod, &obj_out,
                                    llvm::FileType::ObjectFile)
                })?;
                timeline.record("obj");
            } else if asm_to_obj {
                let assembly = cgcx.output_filenames.temp_path(OutputType::Assembly, module_name);
                run_assembler(cgcx, diag_handler, &assembly, &obj_out);
                timeline.record("asm_to_obj");

                if !config.emit_asm && !cgcx.save_temps {
                    drop(fs::remove_file(&assembly));
                }
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
    }
    Ok(module.into_compiled_module(config.emit_obj,
                                   config.emit_bc,
                                   config.emit_bc_compressed,
                                   &cgcx.output_filenames))
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
/// recognize what's going on. For us though we just always throw in an empty
/// cmdline section.
///
/// Furthermore debug/O1 builds don't actually embed bitcode but rather just
/// embed an empty section.
///
/// Basically all of this is us attempting to follow in the footsteps of clang
/// on iOS. See #35968 for lots more info.
unsafe fn embed_bitcode(cgcx: &CodegenContext,
                        llcx: &llvm::Context,
                        llmod: &llvm::Module,
                        bitcode: Option<&[u8]>) {
    let llconst = C_bytes_in_context(llcx, bitcode.unwrap_or(&[]));
    let llglobal = llvm::LLVMAddGlobal(
        llmod,
        val_ty(llconst),
        "rustc.embedded.module\0".as_ptr() as *const _,
    );
    llvm::LLVMSetInitializer(llglobal, llconst);

    let is_apple = cgcx.opts.target_triple.triple().contains("-ios") ||
                   cgcx.opts.target_triple.triple().contains("-darwin");

    let section = if is_apple {
        "__LLVM,__bitcode\0"
    } else {
        ".llvmbc\0"
    };
    llvm::LLVMSetSection(llglobal, section.as_ptr() as *const _);
    llvm::LLVMRustSetLinkage(llglobal, llvm::Linkage::PrivateLinkage);
    llvm::LLVMSetGlobalConstant(llglobal, llvm::True);

    let llconst = C_bytes_in_context(llcx, &[]);
    let llglobal = llvm::LLVMAddGlobal(
        llmod,
        val_ty(llconst),
        "rustc.embedded.cmdline\0".as_ptr() as *const _,
    );
    llvm::LLVMSetInitializer(llglobal, llconst);
    let section = if  is_apple {
        "__LLVM,__cmdline\0"
    } else {
        ".llvmcmd\0"
    };
    llvm::LLVMSetSection(llglobal, section.as_ptr() as *const _);
    llvm::LLVMRustSetLinkage(llglobal, llvm::Linkage::PrivateLinkage);
}

pub(crate) struct CompiledModules {
    pub modules: Vec<CompiledModule>,
    pub metadata_module: CompiledModule,
    pub allocator_module: Option<CompiledModule>,
}

fn need_crate_bitcode_for_rlib(sess: &Session) -> bool {
    sess.crate_types.borrow().contains(&config::CrateType::Rlib) &&
    sess.opts.output_types.contains_key(&OutputType::Exe)
}

fn need_pre_thin_lto_bitcode_for_incr_comp(sess: &Session) -> bool {
    if sess.opts.incremental.is_none() {
        return false
    }

    match sess.lto() {
        Lto::Fat |
        Lto::No => false,
        Lto::Thin |
        Lto::ThinLocal => true,
    }
}

pub fn start_async_codegen(tcx: TyCtxt,
                               time_graph: Option<TimeGraph>,
                               metadata: EncodedMetadata,
                               coordinator_receive: Receiver<Box<dyn Any + Send>>,
                               total_cgus: usize)
                               -> OngoingCodegen {
    let sess = tcx.sess;
    let crate_name = tcx.crate_name(LOCAL_CRATE);
    let crate_hash = tcx.crate_hash(LOCAL_CRATE);
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

    let linker_info = LinkerInfo::new(tcx);
    let crate_info = CrateInfo::new(tcx);

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

    modules_config.pgo_gen = sess.opts.debugging_opts.pgo_gen.clone();
    modules_config.pgo_use = sess.opts.debugging_opts.pgo_use.clone();

    modules_config.opt_level = Some(get_llvm_opt_level(sess.opts.optimize));
    modules_config.opt_size = Some(get_llvm_opt_size(sess.opts.optimize));

    // Save all versions of the bytecode if we're saving our temporaries.
    if sess.opts.cg.save_temps {
        modules_config.emit_no_opt_bc = true;
        modules_config.emit_pre_thin_lto_bc = true;
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

    modules_config.emit_pre_thin_lto_bc =
        need_pre_thin_lto_bitcode_for_incr_comp(sess);

    modules_config.no_integrated_as = tcx.sess.opts.cg.no_integrated_as ||
        tcx.sess.target.target.options.no_integrated_as;

    for output_type in sess.opts.output_types.keys() {
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

    let (shared_emitter, shared_emitter_main) = SharedEmitter::new();
    let (codegen_worker_send, codegen_worker_receive) = channel();

    let coordinator_thread = start_executing_work(tcx,
                                                  &crate_info,
                                                  shared_emitter,
                                                  codegen_worker_send,
                                                  coordinator_receive,
                                                  total_cgus,
                                                  sess.jobserver.clone(),
                                                  time_graph.clone(),
                                                  Arc::new(modules_config),
                                                  Arc::new(metadata_config),
                                                  Arc::new(allocator_config));

    OngoingCodegen {
        crate_name,
        crate_hash,
        metadata,
        windows_subsystem,
        linker_info,
        crate_info,

        time_graph,
        coordinator_send: tcx.tx_to_llvm_workers.lock().clone(),
        codegen_worker_receive,
        shared_emitter_main,
        future: coordinator_thread,
        output_filenames: tcx.output_filenames(LOCAL_CRATE),
    }
}

fn copy_all_cgu_workproducts_to_incr_comp_cache_dir(
    sess: &Session,
    compiled_modules: &CompiledModules,
) -> FxHashMap<WorkProductId, WorkProduct> {
    let mut work_products = FxHashMap::default();

    if sess.opts.incremental.is_none() {
        return work_products;
    }

    for module in compiled_modules.modules.iter().filter(|m| m.kind == ModuleKind::Regular) {
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

        if let Some((id, product)) =
                copy_cgu_workproducts_to_incr_comp_cache_dir(sess, &module.name, &files) {
            work_products.insert(id, product);
        }
    }

    work_products
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

pub(crate) fn dump_incremental_data(_codegen_results: &CodegenResults) {
    // FIXME(mw): This does not work at the moment because the situation has
    //            become more complicated due to incremental LTO. Now a CGU
    //            can have more than two caching states.
    // println!("[incremental] Re-using {} out of {} modules",
    //           codegen_results.modules.iter().filter(|m| m.pre_existing).count(),
    //           codegen_results.modules.len());
}

enum WorkItem {
    /// Optimize a newly codegened, totally unoptimized module.
    Optimize(ModuleCodegen),
    /// Copy the post-LTO artifacts from the incremental cache to the output
    /// directory.
    CopyPostLtoArtifacts(CachedModuleCodegen),
    /// Perform (Thin)LTO on the given module.
    LTO(lto::LtoModuleCodegen),
}

impl WorkItem {
    fn module_kind(&self) -> ModuleKind {
        match *self {
            WorkItem::Optimize(ref m) => m.kind,
            WorkItem::CopyPostLtoArtifacts(_) |
            WorkItem::LTO(_) => ModuleKind::Regular,
        }
    }

    fn name(&self) -> String {
        match *self {
            WorkItem::Optimize(ref m) => format!("optimize: {}", m.name),
            WorkItem::CopyPostLtoArtifacts(ref m) => format!("copy post LTO artifacts: {}", m.name),
            WorkItem::LTO(ref m) => format!("lto: {}", m.name()),
        }
    }
}

enum WorkItemResult {
    Compiled(CompiledModule),
    NeedsLTO(ModuleCodegen),
}

fn execute_work_item(cgcx: &CodegenContext,
                     work_item: WorkItem,
                     timeline: &mut Timeline)
    -> Result<WorkItemResult, FatalError>
{
    let module_config = cgcx.config(work_item.module_kind());

    match work_item {
        WorkItem::Optimize(module) => {
            execute_optimize_work_item(cgcx, module, module_config, timeline)
        }
        WorkItem::CopyPostLtoArtifacts(module) => {
            execute_copy_from_cache_work_item(cgcx, module, module_config, timeline)
        }
        WorkItem::LTO(module) => {
            execute_lto_work_item(cgcx, module, module_config, timeline)
        }
    }
}

fn execute_optimize_work_item(cgcx: &CodegenContext,
                              module: ModuleCodegen,
                              module_config: &ModuleConfig,
                              timeline: &mut Timeline)
    -> Result<WorkItemResult, FatalError>
{
    let diag_handler = cgcx.create_diag_handler();

    unsafe {
        optimize(cgcx, &diag_handler, &module, module_config, timeline)?;
    }

    let linker_does_lto = cgcx.opts.debugging_opts.cross_lang_lto.enabled();

    // After we've done the initial round of optimizations we need to
    // decide whether to synchronously codegen this module or ship it
    // back to the coordinator thread for further LTO processing (which
    // has to wait for all the initial modules to be optimized).
    //
    // Here we dispatch based on the `cgcx.lto` and kind of module we're
    // codegenning...
    let needs_lto = match cgcx.lto {
        Lto::No => false,

        // If the linker does LTO, we don't have to do it. Note that we
        // keep doing full LTO, if it is requested, as not to break the
        // assumption that the output will be a single module.
        Lto::Thin | Lto::ThinLocal if linker_does_lto => false,

        // Here we've got a full crate graph LTO requested. We ignore
        // this, however, if the crate type is only an rlib as there's
        // no full crate graph to process, that'll happen later.
        //
        // This use case currently comes up primarily for targets that
        // require LTO so the request for LTO is always unconditionally
        // passed down to the backend, but we don't actually want to do
        // anything about it yet until we've got a final product.
        Lto::Fat | Lto::Thin => {
            cgcx.crate_types.len() != 1 ||
                cgcx.crate_types[0] != config::CrateType::Rlib
        }

        // When we're automatically doing ThinLTO for multi-codegen-unit
        // builds we don't actually want to LTO the allocator modules if
        // it shows up. This is due to various linker shenanigans that
        // we'll encounter later.
        //
        // Additionally here's where we also factor in the current LLVM
        // version. If it doesn't support ThinLTO we skip this.
        Lto::ThinLocal => {
            module.kind != ModuleKind::Allocator &&
                unsafe { llvm::LLVMRustThinLTOAvailable() }
        }
    };

    // Metadata modules never participate in LTO regardless of the lto
    // settings.
    let needs_lto = needs_lto && module.kind != ModuleKind::Metadata;

    if needs_lto {
        Ok(WorkItemResult::NeedsLTO(module))
    } else {
        let module = unsafe {
            codegen(cgcx, &diag_handler, module, module_config, timeline)?
        };
        Ok(WorkItemResult::Compiled(module))
    }
}

fn execute_copy_from_cache_work_item(cgcx: &CodegenContext,
                                     module: CachedModuleCodegen,
                                     module_config: &ModuleConfig,
                                     _: &mut Timeline)
    -> Result<WorkItemResult, FatalError>
{
    let incr_comp_session_dir = cgcx.incr_comp_session_dir
                                    .as_ref()
                                    .unwrap();
    let mut object = None;
    let mut bytecode = None;
    let mut bytecode_compressed = None;
    for (kind, saved_file) in &module.source.saved_files {
        let obj_out = match kind {
            WorkProductFileKind::Object => {
                let path = cgcx.output_filenames.temp_path(OutputType::Object,
                                                           Some(&module.name));
                object = Some(path.clone());
                path
            }
            WorkProductFileKind::Bytecode => {
                let path = cgcx.output_filenames.temp_path(OutputType::Bitcode,
                                                           Some(&module.name));
                bytecode = Some(path.clone());
                path
            }
            WorkProductFileKind::BytecodeCompressed => {
                let path = cgcx.output_filenames.temp_path(OutputType::Bitcode,
                                                           Some(&module.name))
                    .with_extension(RLIB_BYTECODE_EXTENSION);
                bytecode_compressed = Some(path.clone());
                path
            }
        };
        let source_file = in_incr_comp_dir(&incr_comp_session_dir,
                                           &saved_file);
        debug!("copying pre-existing module `{}` from {:?} to {}",
               module.name,
               source_file,
               obj_out.display());
        match link_or_copy(&source_file, &obj_out) {
            Ok(_) => { }
            Err(err) => {
                let diag_handler = cgcx.create_diag_handler();
                diag_handler.err(&format!("unable to copy {} to {}: {}",
                                          source_file.display(),
                                          obj_out.display(),
                                          err));
            }
        }
    }

    assert_eq!(object.is_some(), module_config.emit_obj);
    assert_eq!(bytecode.is_some(), module_config.emit_bc);
    assert_eq!(bytecode_compressed.is_some(), module_config.emit_bc_compressed);

    Ok(WorkItemResult::Compiled(CompiledModule {
        name: module.name,
        kind: ModuleKind::Regular,
        object,
        bytecode,
        bytecode_compressed,
    }))
}

fn execute_lto_work_item(cgcx: &CodegenContext,
                         mut module: lto::LtoModuleCodegen,
                         module_config: &ModuleConfig,
                         timeline: &mut Timeline)
    -> Result<WorkItemResult, FatalError>
{
    let diag_handler = cgcx.create_diag_handler();

    unsafe {
        let module = module.optimize(cgcx, timeline)?;
        let module = codegen(cgcx, &diag_handler, module, module_config, timeline)?;
        Ok(WorkItemResult::Compiled(module))
    }
}

enum Message {
    Token(io::Result<Acquired>),
    NeedsLTO {
        result: ModuleCodegen,
        worker_id: usize,
    },
    Done {
        result: Result<CompiledModule, ()>,
        worker_id: usize,
    },
    CodegenDone {
        llvm_work_item: WorkItem,
        cost: u64,
    },
    AddImportOnlyModule {
        module_data: SerializedModule,
        work_product: WorkProduct,
    },
    CodegenComplete,
    CodegenItem,
}

struct Diagnostic {
    msg: String,
    code: Option<DiagnosticId>,
    lvl: Level,
}

#[derive(PartialEq, Clone, Copy, Debug)]
enum MainThreadWorkerState {
    Idle,
    Codegenning,
    LLVMing,
}

fn start_executing_work(tcx: TyCtxt,
                        crate_info: &CrateInfo,
                        shared_emitter: SharedEmitter,
                        codegen_worker_send: Sender<Message>,
                        coordinator_receive: Receiver<Box<dyn Any + Send>>,
                        total_cgus: usize,
                        jobserver: Client,
                        time_graph: Option<TimeGraph>,
                        modules_config: Arc<ModuleConfig>,
                        metadata_config: Arc<ModuleConfig>,
                        allocator_config: Arc<ModuleConfig>)
                        -> thread::JoinHandle<Result<CompiledModules, ()>> {
    let coordinator_send = tcx.tx_to_llvm_workers.lock().clone();
    let sess = tcx.sess;

    // Compute the set of symbols we need to retain when doing LTO (if we need to)
    let exported_symbols = {
        let mut exported_symbols = FxHashMap();

        let copy_symbols = |cnum| {
            let symbols = tcx.exported_symbols(cnum)
                             .iter()
                             .map(|&(s, lvl)| (s.symbol_name(tcx).to_string(), lvl))
                             .collect();
            Arc::new(symbols)
        };

        match sess.lto() {
            Lto::No => None,
            Lto::ThinLocal => {
                exported_symbols.insert(LOCAL_CRATE, copy_symbols(LOCAL_CRATE));
                Some(Arc::new(exported_symbols))
            }
            Lto::Fat | Lto::Thin => {
                exported_symbols.insert(LOCAL_CRATE, copy_symbols(LOCAL_CRATE));
                for &cnum in tcx.crates().iter() {
                    exported_symbols.insert(cnum, copy_symbols(cnum));
                }
                Some(Arc::new(exported_symbols))
            }
        }
    };

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

    let assembler_cmd = if modules_config.no_integrated_as {
        // HACK: currently we use linker (gcc) as our assembler
        let (linker, flavor) = link::linker_and_flavor(sess);

        let (name, mut cmd) = get_linker(sess, &linker, flavor);
        cmd.args(&sess.target.target.options.asm_args);
        Some(Arc::new(AssemblerCommand {
            name,
            cmd,
        }))
    } else {
        None
    };

    let cgcx = CodegenContext {
        crate_types: sess.crate_types.borrow().clone(),
        each_linked_rlib_for_lto,
        lto: sess.lto(),
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
        cgu_reuse_tracker: sess.cgu_reuse_tracker.clone(),
        coordinator_send,
        diag_emitter: shared_emitter.clone(),
        time_graph,
        output_filenames: tcx.output_filenames(LOCAL_CRATE),
        regular_module_config: modules_config,
        metadata_module_config: metadata_config,
        allocator_module_config: allocator_config,
        tm_factory: target_machine_factory(tcx.sess, false),
        total_cgus,
        msvc_imps_needed: msvc_imps_needed(tcx),
        target_pointer_width: tcx.sess.target.target.target_pointer_width.clone(),
        debuginfo: tcx.sess.opts.debuginfo,
        assembler_cmd,
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
    // - Codegen can only happen on the main thread.
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
    // Since the main thread will often be busy doing codegen work, error
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
    // The job of the main thread is to codegen CGUs into LLVM work package
    // (since the main thread is the only thread that can do this). The main
    // thread will block until it receives a message from the coordinator, upon
    // which it will codegen one CGU, send it to the coordinator and block
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
    // Tokens/LLVM worker threads. However, since codegen usual is faster
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
    // modules have been codegened and optimized we hand this to the `lto`
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
        // through codegen and LLVM.
        let mut compiled_modules = vec![];
        let mut compiled_metadata_module = None;
        let mut compiled_allocator_module = None;
        let mut needs_lto = Vec::new();
        let mut lto_import_only_modules = Vec::new();
        let mut started_lto = false;

        // This flag tracks whether all items have gone through codegens
        let mut codegen_done = false;

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
        while !codegen_done ||
              work_items.len() > 0 ||
              running > 0 ||
              needs_lto.len() > 0 ||
              lto_import_only_modules.len() > 0 ||
              main_thread_worker_state != MainThreadWorkerState::Idle {

            // While there are still CGUs to be codegened, the coordinator has
            // to decide how to utilize the compiler processes implicit Token:
            // For codegenning more CGU or for running them through LLVM.
            if !codegen_done {
                if main_thread_worker_state == MainThreadWorkerState::Idle {
                    if !queue_full_enough(work_items.len(), running, max_workers) {
                        // The queue is not full enough, codegen more items:
                        if let Err(_) = codegen_worker_send.send(Message::CodegenItem) {
                            panic!("Could not send Message::CodegenItem to main thread")
                        }
                        main_thread_worker_state = MainThreadWorkerState::Codegenning;
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
                        maybe_start_llvm_timer(cgcx.config(item.module_kind()),
                                               &mut llvm_start_time);
                        main_thread_worker_state = MainThreadWorkerState::LLVMing;
                        spawn_work(cgcx, item);
                    }
                }
            } else {
                // If we've finished everything related to normal codegen
                // then it must be the case that we've got some LTO work to do.
                // Perform the serial work here of figuring out what we're
                // going to LTO and then push a bunch of work items onto our
                // queue to do LTO
                if work_items.len() == 0 &&
                   running == 0 &&
                   main_thread_worker_state == MainThreadWorkerState::Idle {
                    assert!(!started_lto);
                    assert!(needs_lto.len() + lto_import_only_modules.len() > 0);
                    started_lto = true;
                    let modules = mem::replace(&mut needs_lto, Vec::new());
                    let import_only_modules =
                        mem::replace(&mut lto_import_only_modules, Vec::new());
                    for (work, cost) in generate_lto_work(&cgcx, modules, import_only_modules) {
                        let insertion_index = work_items
                            .binary_search_by_key(&cost, |&(_, cost)| cost)
                            .unwrap_or_else(|e| e);
                        work_items.insert(insertion_index, (work, cost));
                        if !cgcx.opts.debugging_opts.no_parallel_llvm {
                            helper.request_token();
                        }
                    }
                }

                // In this branch, we know that everything has been codegened,
                // so it's just a matter of determining whether the implicit
                // Token is free to use for LLVM work.
                match main_thread_worker_state {
                    MainThreadWorkerState::Idle => {
                        if let Some((item, _)) = work_items.pop() {
                            let cgcx = CodegenContext {
                                worker: get_worker_id(&mut free_worker_ids),
                                .. cgcx.clone()
                            };
                            maybe_start_llvm_timer(cgcx.config(item.module_kind()),
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
                    MainThreadWorkerState::Codegenning => {
                        bug!("codegen worker should not be codegenning after \
                              codegen was already completed")
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

                maybe_start_llvm_timer(cgcx.config(item.module_kind()),
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
                                // to react to codegen demand.
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

                Message::CodegenDone { llvm_work_item, cost } => {
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

                    if !cgcx.opts.debugging_opts.no_parallel_llvm {
                        helper.request_token();
                    }
                    assert_eq!(main_thread_worker_state,
                               MainThreadWorkerState::Codegenning);
                    main_thread_worker_state = MainThreadWorkerState::Idle;
                }

                Message::CodegenComplete => {
                    codegen_done = true;
                    assert_eq!(main_thread_worker_state,
                               MainThreadWorkerState::Codegenning);
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
                Message::AddImportOnlyModule { module_data, work_product } => {
                    assert!(!started_lto);
                    assert!(!codegen_done);
                    assert_eq!(main_thread_worker_state,
                               MainThreadWorkerState::Codegenning);
                    lto_import_only_modules.push((module_data, work_product));
                    main_thread_worker_state = MainThreadWorkerState::Idle;
                }
                Message::Done { result: Err(()), worker_id: _ } => {
                    shared_emitter.fatal("aborting due to worker thread failure");
                    // Exit the coordinator thread
                    return Err(())
                }
                Message::CodegenItem => {
                    bug!("the coordinator should not receive codegen requests")
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
    // queue so that the main thread can do LLVM work instead of codegen
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

pub const CODEGEN_WORKER_ID: usize = ::std::usize::MAX;
pub const CODEGEN_WORKER_TIMELINE: time_graph::TimelineId =
    time_graph::TimelineId(CODEGEN_WORKER_ID);
pub const CODEGEN_WORK_PACKAGE_KIND: time_graph::WorkPackageKind =
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
            coordinator_send: Sender<Box<dyn Any + Send>>,
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

pub fn run_assembler(cgcx: &CodegenContext, handler: &Handler, assembly: &Path, object: &Path) {
    let assembler = cgcx.assembler_cmd
        .as_ref()
        .expect("cgcx.assembler_cmd is missing?");

    let pname = &assembler.name;
    let mut cmd = assembler.cmd.clone();
    cmd.arg("-c").arg("-o").arg(object).arg(assembly);
    debug!("{:?}", cmd);

    match cmd.output() {
        Ok(prog) => {
            if !prog.status.success() {
                let mut note = prog.stderr.clone();
                note.extend_from_slice(&prog.stdout);

                handler.struct_err(&format!("linking with `{}` failed: {}",
                                            pname.display(),
                                            prog.status))
                    .note(&format!("{:?}", &cmd))
                    .note(str::from_utf8(&note[..]).unwrap())
                    .emit();
                handler.abort_if_errors();
            }
        },
        Err(e) => {
            handler.err(&format!("could not exec the linker `{}`: {}", pname.display(), e));
            handler.abort_if_errors();
        }
    }
}

pub unsafe fn with_llvm_pmb(llmod: &llvm::Module,
                            config: &ModuleConfig,
                            opt_level: llvm::CodeGenOptLevel,
                            prepare_for_thin_lto: bool,
                            f: &mut dyn FnMut(&llvm::PassManagerBuilder)) {
    use std::ptr;

    // Create the PassManagerBuilder for LLVM. We configure it with
    // reasonable defaults and prepare it to actually populate the pass
    // manager.
    let builder = llvm::LLVMPassManagerBuilderCreate();
    let opt_size = config.opt_size.unwrap_or(llvm::CodeGenOptSizeNone);
    let inline_threshold = config.inline_threshold;

    let pgo_gen_path = config.pgo_gen.as_ref().map(|s| {
        let s = if s.is_empty() { "default_%m.profraw" } else { s };
        CString::new(s.as_bytes()).unwrap()
    });

    let pgo_use_path = if config.pgo_use.is_empty() {
        None
    } else {
        Some(CString::new(config.pgo_use.as_bytes()).unwrap())
    };

    llvm::LLVMRustConfigurePassManagerBuilder(
        builder,
        opt_level,
        config.merge_functions,
        config.vectorize_slp,
        config.vectorize_loop,
        prepare_for_thin_lto,
        pgo_gen_path.as_ref().map_or(ptr::null(), |s| s.as_ptr()),
        pgo_use_path.as_ref().map_or(ptr::null(), |s| s.as_ptr()),
    );

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

pub struct OngoingCodegen {
    crate_name: Symbol,
    crate_hash: Svh,
    metadata: EncodedMetadata,
    windows_subsystem: Option<String>,
    linker_info: LinkerInfo,
    crate_info: CrateInfo,
    time_graph: Option<TimeGraph>,
    coordinator_send: Sender<Box<dyn Any + Send>>,
    codegen_worker_receive: Receiver<Message>,
    shared_emitter_main: SharedEmitterMain,
    future: thread::JoinHandle<Result<CompiledModules, ()>>,
    output_filenames: Arc<OutputFilenames>,
}

impl OngoingCodegen {
    pub(crate) fn join(
        self,
        sess: &Session
    ) -> (CodegenResults, FxHashMap<WorkProductId, WorkProduct>) {
        self.shared_emitter_main.check(sess, true);
        let compiled_modules = match self.future.join() {
            Ok(Ok(compiled_modules)) => compiled_modules,
            Ok(Err(())) => {
                sess.abort_if_errors();
                panic!("expected abort due to worker thread errors")
            },
            Err(_) => {
                sess.fatal("Error during codegen/LLVM phase.");
            }
        };

        sess.cgu_reuse_tracker.check_expected_reuse(sess);

        sess.abort_if_errors();

        if let Some(time_graph) = self.time_graph {
            time_graph.dump(&format!("{}-timings", self.crate_name));
        }

        let work_products =
            copy_all_cgu_workproducts_to_incr_comp_cache_dir(sess,
                                                             &compiled_modules);
        produce_final_output_artifacts(sess,
                                       &compiled_modules,
                                       &self.output_filenames);

        // FIXME: time_llvm_passes support - does this use a global context or
        // something?
        if sess.codegen_units() == 1 && sess.time_llvm_passes() {
            unsafe { llvm::LLVMRustPrintPassTimings(); }
        }

        (CodegenResults {
            crate_name: self.crate_name,
            crate_hash: self.crate_hash,
            metadata: self.metadata,
            windows_subsystem: self.windows_subsystem,
            linker_info: self.linker_info,
            crate_info: self.crate_info,

            modules: compiled_modules.modules,
            allocator_module: compiled_modules.allocator_module,
            metadata_module: compiled_modules.metadata_module,
        }, work_products)
    }

    pub(crate) fn submit_pre_codegened_module_to_llvm(&self,
                                                       tcx: TyCtxt,
                                                       module: ModuleCodegen) {
        self.wait_for_signal_to_codegen_item();
        self.check_for_errors(tcx.sess);

        // These are generally cheap and won't through off scheduling.
        let cost = 0;
        submit_codegened_module_to_llvm(tcx, module, cost);
    }

    pub fn codegen_finished(&self, tcx: TyCtxt) {
        self.wait_for_signal_to_codegen_item();
        self.check_for_errors(tcx.sess);
        drop(self.coordinator_send.send(Box::new(Message::CodegenComplete)));
    }

    pub fn check_for_errors(&self, sess: &Session) {
        self.shared_emitter_main.check(sess, false);
    }

    pub fn wait_for_signal_to_codegen_item(&self) {
        match self.codegen_worker_receive.recv() {
            Ok(Message::CodegenItem) => {
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

pub(crate) fn submit_codegened_module_to_llvm(tcx: TyCtxt,
                                              module: ModuleCodegen,
                                              cost: u64) {
    let llvm_work_item = WorkItem::Optimize(module);
    drop(tcx.tx_to_llvm_workers.lock().send(Box::new(Message::CodegenDone {
        llvm_work_item,
        cost,
    })));
}

pub(crate) fn submit_post_lto_module_to_llvm(tcx: TyCtxt,
                                             module: CachedModuleCodegen) {
    let llvm_work_item = WorkItem::CopyPostLtoArtifacts(module);
    drop(tcx.tx_to_llvm_workers.lock().send(Box::new(Message::CodegenDone {
        llvm_work_item,
        cost: 0,
    })));
}

pub(crate) fn submit_pre_lto_module_to_llvm(tcx: TyCtxt,
                                            module: CachedModuleCodegen) {
    let filename = pre_lto_bitcode_filename(&module.name);
    let bc_path = in_incr_comp_dir_sess(tcx.sess, &filename);
    let file = fs::File::open(&bc_path).unwrap_or_else(|e| {
        panic!("failed to open bitcode file `{}`: {}", bc_path.display(), e)
    });

    let mmap = unsafe {
        memmap::Mmap::map(&file).unwrap_or_else(|e| {
            panic!("failed to mmap bitcode file `{}`: {}", bc_path.display(), e)
        })
    };

    // Schedule the module to be loaded
    drop(tcx.tx_to_llvm_workers.lock().send(Box::new(Message::AddImportOnlyModule {
        module_data: SerializedModule::FromUncompressedFile(mmap),
        work_product: module.source,
    })));
}

pub(super) fn pre_lto_bitcode_filename(module_name: &str) -> String {
    format!("{}.{}", module_name, PRE_THIN_LTO_BC_EXT)
}

fn msvc_imps_needed(tcx: TyCtxt) -> bool {
    // This should never be true (because it's not supported). If it is true,
    // something is wrong with commandline arg validation.
    assert!(!(tcx.sess.opts.debugging_opts.cross_lang_lto.enabled() &&
              tcx.sess.target.target.options.is_like_msvc &&
              tcx.sess.opts.cg.prefer_dynamic));

    tcx.sess.target.target.options.is_like_msvc &&
        tcx.sess.crate_types.borrow().iter().any(|ct| *ct == config::CrateType::Rlib) &&
    // ThinLTO can't handle this workaround in all cases, so we don't
    // emit the `__imp_` symbols. Instead we make them unnecessary by disallowing
    // dynamic linking when cross-language LTO is enabled.
    !tcx.sess.opts.debugging_opts.cross_lang_lto.enabled()
}

// Create a `__imp_<symbol> = &symbol` global for every public static `symbol`.
// This is required to satisfy `dllimport` references to static data in .rlibs
// when using MSVC linker.  We do this only for data, as linker can fix up
// code references on its own.
// See #26591, #27438
fn create_msvc_imps(cgcx: &CodegenContext, llcx: &llvm::Context, llmod: &llvm::Module) {
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
                                          i8p_ty,
                                          imp_name.as_ptr() as *const _);
            llvm::LLVMSetInitializer(imp, consts::ptrcast(val, i8p_ty));
            llvm::LLVMRustSetLinkage(imp, llvm::Linkage::ExternalLinkage);
        }
    }
}
