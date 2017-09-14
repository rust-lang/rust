// Copyright 2013-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use back::lto;
use back::link::{self, get_linker, remove};
use back::linker::LinkerInfo;
use back::symbol_export::ExportedSymbols;
use rustc_incremental::{save_trans_partition, in_incr_comp_dir};
use rustc::middle::cstore::{LinkMeta, EncodedMetadata};
use rustc::session::config::{self, OutputFilenames, OutputType, OutputTypes, Passes, SomePasses,
                             AllPasses, Sanitizer};
use rustc::session::Session;
use time_graph::{self, TimeGraph};
use llvm;
use llvm::{ModuleRef, TargetMachineRef, PassManagerRef, DiagnosticInfoRef};
use llvm::SMDiagnosticRef;
use {CrateTranslation, ModuleSource, ModuleTranslation, CompiledModule, ModuleKind};
use CrateInfo;
use rustc::hir::def_id::CrateNum;
use rustc::util::common::{time, time_depth, set_time_depth, path2cstr, print_time_passes_entry};
use rustc::util::fs::{link_or_copy, rename_or_copy_remove};
use errors::{self, Handler, Level, DiagnosticBuilder, FatalError};
use errors::emitter::{Emitter};
use syntax::ext::hygiene::Mark;
use syntax_pos::MultiSpan;
use syntax_pos::symbol::Symbol;
use context::{is_pie_binary, get_reloc_model};
use jobserver::{Client, Acquired};
use rustc_demangle;

use std::ffi::CString;
use std::fmt;
use std::fs;
use std::io;
use std::io::Write;
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
                             sess.opts
                                 .cg
                                 .code_model));
            sess.abort_if_errors();
            bug!();
        }
    };

    let triple = &sess.target.target.llvm_target;

    let tm = unsafe {
        let triple = CString::new(triple.as_bytes()).unwrap();
        let cpu = match sess.opts.cg.target_cpu {
            Some(ref s) => &**s,
            None => &*sess.target.target.options.cpu
        };
        let cpu = CString::new(cpu.as_bytes()).unwrap();
        let features = CString::new(target_feature(sess).as_bytes()).unwrap();
        llvm::LLVMRustCreateTargetMachine(
            triple.as_ptr(), cpu.as_ptr(), features.as_ptr(),
            code_model,
            reloc_model,
            opt_level,
            use_softfp,
            is_pie_binary(sess),
            ffunction_sections,
            fdata_sections,
        )
    };

    if tm.is_null() {
        let msg = format!("Could not create LLVM TargetMachine for triple: {}",
                          triple);
        panic!(llvm_err(sess.diagnostic(), msg));
    } else {
        return tm;
    };
}


/// Module-specific configuration for `optimize_and_codegen`.
pub struct ModuleConfig {
    /// LLVM TargetMachine to use for codegen.
    tm: TargetMachineRef,
    /// Names of additional optimization passes to run.
    passes: Vec<String>,
    /// Some(level) to optimize at a certain level, or None to run
    /// absolutely no optimizations (used for the metadata module).
    opt_level: Option<llvm::CodeGenOptLevel>,

    /// Some(level) to optimize binary size, or None to not affect program size.
    opt_size: Option<llvm::CodeGenOptSize>,

    // Flags indicating which outputs to produce.
    emit_no_opt_bc: bool,
    emit_bc: bool,
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

unsafe impl Send for ModuleConfig { }

impl ModuleConfig {
    fn new(sess: &Session, passes: Vec<String>) -> ModuleConfig {
        ModuleConfig {
            tm: create_target_machine(sess),
            passes,
            opt_level: None,
            opt_size: None,

            emit_no_opt_bc: false,
            emit_bc: false,
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
        self.no_builtins = no_builtins;
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

    fn clone(&self, sess: &Session) -> ModuleConfig {
        ModuleConfig {
            tm: create_target_machine(sess),
            passes: self.passes.clone(),
            opt_level: self.opt_level,
            opt_size: self.opt_size,

            emit_no_opt_bc: self.emit_no_opt_bc,
            emit_bc: self.emit_bc,
            emit_lto_bc: self.emit_lto_bc,
            emit_ir: self.emit_ir,
            emit_asm: self.emit_asm,
            emit_obj: self.emit_obj,
            obj_is_bitcode: self.obj_is_bitcode,

            no_verify: self.no_verify,
            no_prepopulate_passes: self.no_prepopulate_passes,
            no_builtins: self.no_builtins,
            time_passes: self.time_passes,
            vectorize_loop: self.vectorize_loop,
            vectorize_slp: self.vectorize_slp,
            merge_functions: self.merge_functions,
            inline_threshold: self.inline_threshold,
        }
    }
}

impl Drop for ModuleConfig {
    fn drop(&mut self) {
        unsafe {
            llvm::LLVMRustDisposeTargetMachine(self.tm);
        }
    }
}

/// Additional resources used by optimize_and_codegen (not module specific)
#[derive(Clone)]
pub struct CodegenContext {
    // Resouces needed when running LTO
    pub time_passes: bool,
    pub lto: bool,
    pub no_landing_pads: bool,
    pub exported_symbols: Arc<ExportedSymbols>,
    pub opts: Arc<config::Options>,
    pub crate_types: Vec<config::CrateType>,
    pub each_linked_rlib_for_lto: Vec<(CrateNum, PathBuf)>,
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
    coordinator_send: Sender<Message>,
    // A reference to the TimeGraph so we can register timings. None means that
    // measuring is disabled.
    time_graph: Option<TimeGraph>,
}

impl CodegenContext {
    fn create_diag_handler(&self) -> Handler {
        Handler::with_emitter(true, false, Box::new(self.diag_emitter.clone()))
    }
}

struct HandlerFreeVars<'a> {
    cgcx: &'a CodegenContext,
    diag_handler: &'a Handler,
}

unsafe extern "C" fn report_inline_asm<'a, 'b>(cgcx: &'a CodegenContext,
                                               msg: &'b str,
                                               cookie: c_uint) {
    cgcx.diag_emitter.inline_asm_error(cookie as u32, msg.to_string());
}

unsafe extern "C" fn inline_asm_handler(diag: SMDiagnosticRef,
                                        user: *const c_void,
                                        cookie: c_uint) {
    let HandlerFreeVars { cgcx, .. } = *(user as *const HandlerFreeVars);

    let msg = llvm::build_string(|s| llvm::LLVMRustWriteSMDiagnosticToString(diag, s))
        .expect("non-UTF8 SMDiagnostic");

    report_inline_asm(cgcx, &msg, cookie);
}

unsafe extern "C" fn diagnostic_handler(info: DiagnosticInfoRef, user: *mut c_void) {
    let HandlerFreeVars { cgcx, diag_handler, .. } = *(user as *const HandlerFreeVars);

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
unsafe fn optimize_and_codegen(cgcx: &CodegenContext,
                               diag_handler: &Handler,
                               mtrans: ModuleTranslation,
                               config: ModuleConfig,
                               output_names: OutputFilenames)
    -> Result<CompiledModule, FatalError>
{
    let (llmod, llcx) = match mtrans.source {
        ModuleSource::Translated(ref llvm) => (llvm.llmod, llvm.llcx),
        ModuleSource::Preexisting(_) => {
            bug!("optimize_and_codegen: called with ModuleSource::Preexisting")
        }
    };

    let tm = config.tm;

    let fv = HandlerFreeVars {
        cgcx,
        diag_handler,
    };
    let fv = &fv as *const HandlerFreeVars as *mut c_void;

    llvm::LLVMRustSetInlineAsmDiagnosticHandler(llcx, inline_asm_handler, fv);
    llvm::LLVMContextSetDiagnosticHandler(llcx, diagnostic_handler, fv);

    let module_name = mtrans.name.clone();
    let module_name = Some(&module_name[..]);

    if config.emit_no_opt_bc {
        let out = output_names.temp_path_ext("no-opt.bc", module_name);
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
            with_llvm_pmb(llmod, &config, &mut |b| {
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
        time(config.time_passes, &format!("llvm module passes [{}]", module_name.unwrap()), ||
             llvm::LLVMRunPassManager(mpm, llmod));

        // Deallocate managers that we're now done with
        llvm::LLVMDisposePassManager(fpm);
        llvm::LLVMDisposePassManager(mpm);

        if cgcx.lto {
            time(cgcx.time_passes, "all lto passes", || {
                let temp_no_opt_bc_filename =
                    output_names.temp_path_ext("no-opt.lto.bc", module_name);
                lto::run(cgcx,
                         diag_handler,
                         llmod,
                         tm,
                         &config,
                         &temp_no_opt_bc_filename)
            })?;
            if config.emit_lto_bc {
                let out = output_names.temp_path_ext("lto.bc", module_name);
                let out = path2cstr(&out);
                llvm::LLVMWriteBitcodeToFile(llmod, out.as_ptr());
            }
        }
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

    // Change what we write and cleanup based on whether obj files are
    // just llvm bitcode. In that case write bitcode, and possibly
    // delete the bitcode if it wasn't requested. Don't generate the
    // machine code, instead copy the .o file from the .bc
    let write_bc = config.emit_bc || config.obj_is_bitcode;
    let rm_bc = !config.emit_bc && config.obj_is_bitcode;
    let write_obj = config.emit_obj && !config.obj_is_bitcode;
    let copy_bc_to_obj = config.emit_obj && config.obj_is_bitcode;

    let bc_out = output_names.temp_path(OutputType::Bitcode, module_name);
    let obj_out = output_names.temp_path(OutputType::Object, module_name);

    if write_bc {
        let bc_out_c = path2cstr(&bc_out);
        llvm::LLVMWriteBitcodeToFile(llmod, bc_out_c.as_ptr());
    }

    time(config.time_passes, &format!("codegen passes [{}]", module_name.unwrap()),
         || -> Result<(), FatalError> {
        if config.emit_ir {
            let out = output_names.temp_path(OutputType::LlvmAssembly, module_name);
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
            })
        }

        if config.emit_asm {
            let path = output_names.temp_path(OutputType::Assembly, module_name);

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
        }

        if write_obj {
            with_codegen(tm, llmod, config.no_builtins, |cpm| {
                write_output_file(diag_handler, tm, cpm, llmod, &obj_out,
                                  llvm::FileType::ObjectFile)
            })?;
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

    Ok(mtrans.into_compiled_module(config.emit_obj, config.emit_bc))
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

pub fn start_async_translation(sess: &Session,
                               crate_output: &OutputFilenames,
                               time_graph: Option<TimeGraph>,
                               crate_name: Symbol,
                               link: LinkMeta,
                               metadata: EncodedMetadata,
                               exported_symbols: Arc<ExportedSymbols>,
                               no_builtins: bool,
                               windows_subsystem: Option<String>,
                               linker_info: LinkerInfo,
                               crate_info: CrateInfo,
                               no_integrated_as: bool)
                               -> OngoingCrateTranslation {
    let output_types_override = if no_integrated_as {
        OutputTypes::new(&[(OutputType::Assembly, None)])
    } else {
        sess.opts.output_types.clone()
    };

    // Figure out what we actually need to build.
    let mut modules_config = ModuleConfig::new(sess, sess.opts.cg.passes.clone());
    let mut metadata_config = ModuleConfig::new(sess, vec![]);
    let mut allocator_config = ModuleConfig::new(sess, vec![]);

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

    // Emit bitcode files for the crate if we're emitting an rlib.
    // Whenever an rlib is created, the bitcode is inserted into the
    // archive in order to allow LTO against it.
    if need_crate_bitcode_for_rlib(sess) {
        modules_config.emit_bc = true;
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
    let (coordinator_send, coordinator_receive) = channel();

    let coordinator_thread = start_executing_work(sess,
                                                  &crate_info,
                                                  shared_emitter,
                                                  trans_worker_send,
                                                  coordinator_send.clone(),
                                                  coordinator_receive,
                                                  client,
                                                  time_graph.clone(),
                                                  exported_symbols.clone());
    OngoingCrateTranslation {
        crate_name,
        link,
        metadata,
        windows_subsystem,
        linker_info,
        no_integrated_as,
        crate_info,

        regular_module_config: modules_config,
        metadata_module_config: metadata_config,
        allocator_module_config: allocator_config,

        time_graph,
        output_filenames: crate_output.clone(),
        coordinator_send,
        trans_worker_receive,
        shared_emitter_main,
        future: coordinator_thread
    }
}

fn copy_module_artifacts_into_incr_comp_cache(sess: &Session,
                                              compiled_modules: &CompiledModules,
                                              crate_output: &OutputFilenames) {
    if sess.opts.incremental.is_none() {
        return;
    }

    for module in compiled_modules.modules.iter() {
        let mut files = vec![];

        if module.emit_obj {
            let path = crate_output.temp_path(OutputType::Object, Some(&module.name));
            files.push((OutputType::Object, path));
        }

        if module.emit_bc {
            let path = crate_output.temp_path(OutputType::Bitcode, Some(&module.name));
            files.push((OutputType::Bitcode, path));
        }

        save_trans_partition(sess, &module.name, module.symbol_name_hash, &files);
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
        //  - If we're building an rlib (`needs_crate_bitcode`), then keep
        //    it.
        //  - If the user requested bitcode (`user_wants_bitcode`), and
        //    codegen_units > 1, then keep it.
        //  - If the user requested bitcode but codegen_units == 1, then we
        //    can toss .#module-name#.bc because we copied it to .bc earlier.
        //  - If we're not building an rlib and the user didn't request
        //    bitcode, then delete .#module-name#.bc.
        // If you change how this works, also update back::link::link_rlib,
        // where .#module-name#.bc files are (maybe) deleted after making an
        // rlib.
        let needs_crate_bitcode = need_crate_bitcode_for_rlib(sess);
        let needs_crate_object = crate_output.outputs.contains_key(&OutputType::Exe);

        let keep_numbered_bitcode = needs_crate_bitcode ||
                (user_wants_bitcode && sess.opts.cg.codegen_units > 1);

        let keep_numbered_objects = needs_crate_object ||
                (user_wants_objects && sess.opts.cg.codegen_units > 1);

        for module in compiled_modules.modules.iter() {
            let module_name = Some(&module.name[..]);

            if module.emit_obj && !keep_numbered_objects {
                let path = crate_output.temp_path(OutputType::Object, module_name);
                remove(sess, &path);
            }

            if module.emit_bc && !keep_numbered_bitcode {
                let path = crate_output.temp_path(OutputType::Bitcode, module_name);
                remove(sess, &path);
            }
        }

        if compiled_modules.metadata_module.emit_bc && !user_wants_bitcode {
            let path = crate_output.temp_path(OutputType::Bitcode,
                                              Some(&compiled_modules.metadata_module.name));
            remove(sess, &path);
        }

        if let Some(ref allocator_module) = compiled_modules.allocator_module {
            if allocator_module.emit_bc && !user_wants_bitcode {
                let path = crate_output.temp_path(OutputType::Bitcode,
                                                  Some(&allocator_module.name));
                remove(sess, &path);
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
    let mut reuse = 0;
    for mtrans in trans.modules.iter() {
        if mtrans.pre_existing {
            reuse += 1;
        }
    }
    eprintln!("incremental: re-using {} out of {} modules", reuse, trans.modules.len());
}

struct WorkItem {
    mtrans: ModuleTranslation,
    config: ModuleConfig,
    output_names: OutputFilenames
}

impl fmt::Debug for WorkItem {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "WorkItem({})", self.mtrans.name)
    }
}

fn build_work_item(mtrans: ModuleTranslation,
                   config: ModuleConfig,
                   output_names: OutputFilenames)
                   -> WorkItem
{
    WorkItem {
        mtrans,
        config,
        output_names,
    }
}

fn execute_work_item(cgcx: &CodegenContext, work_item: WorkItem)
    -> Result<CompiledModule, FatalError>
{
    let diag_handler = cgcx.create_diag_handler();
    let module_name = work_item.mtrans.name.clone();

    let pre_existing = match work_item.mtrans.source {
        ModuleSource::Translated(_) => None,
        ModuleSource::Preexisting(ref wp) => Some(wp.clone()),
    };

    if let Some(wp) = pre_existing {
        let incr_comp_session_dir = cgcx.incr_comp_session_dir
                                        .as_ref()
                                        .unwrap();
        let name = &work_item.mtrans.name;
        for (kind, saved_file) in wp.saved_files {
            let obj_out = work_item.output_names.temp_path(kind, Some(name));
            let source_file = in_incr_comp_dir(&incr_comp_session_dir,
                                               &saved_file);
            debug!("copying pre-existing module `{}` from {:?} to {}",
                   work_item.mtrans.name,
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

        Ok(CompiledModule {
            name: module_name,
            kind: ModuleKind::Regular,
            pre_existing: true,
            symbol_name_hash: work_item.mtrans.symbol_name_hash,
            emit_bc: work_item.config.emit_bc,
            emit_obj: work_item.config.emit_obj,
        })
    } else {
        debug!("llvm-optimizing {:?}", module_name);

        unsafe {
            optimize_and_codegen(cgcx,
                                 &diag_handler,
                                 work_item.mtrans,
                                 work_item.config,
                                 work_item.output_names)
        }
    }
}

#[derive(Debug)]
enum Message {
    Token(io::Result<Acquired>),
    Done {
        result: Result<CompiledModule, ()>,
        worker_id: usize,
    },
    TranslationDone {
        llvm_work_item: WorkItem,
        cost: u64,
        is_last: bool,
    },
    TranslateItem,
}

struct Diagnostic {
    msg: String,
    code: Option<String>,
    lvl: Level,
}

#[derive(PartialEq, Clone, Copy, Debug)]
enum MainThreadWorkerState {
    Idle,
    Translating,
    LLVMing,
}

fn start_executing_work(sess: &Session,
                        crate_info: &CrateInfo,
                        shared_emitter: SharedEmitter,
                        trans_worker_send: Sender<Message>,
                        coordinator_send: Sender<Message>,
                        coordinator_receive: Receiver<Message>,
                        jobserver: Client,
                        time_graph: Option<TimeGraph>,
                        exported_symbols: Arc<ExportedSymbols>)
                        -> thread::JoinHandle<CompiledModules> {
    // First up, convert our jobserver into a helper thread so we can use normal
    // mpsc channels to manage our messages and such. Once we've got the helper
    // thread then request `n-1` tokens because all of our work items are ready
    // to go.
    //
    // Note that the `n-1` is here because we ourselves have a token (our
    // process) and we'll use that token to execute at least one unit of work.
    //
    // After we've requested all these tokens then we'll, when we can, get
    // tokens on `rx` above which will get managed in the main loop below.
    let coordinator_send2 = coordinator_send.clone();
    let helper = jobserver.into_helper_thread(move |token| {
        drop(coordinator_send2.send(Message::Token(token)));
    }).expect("failed to spawn helper thread");

    let mut each_linked_rlib_for_lto = Vec::new();
    drop(link::each_linked_rlib(sess, crate_info, &mut |cnum, path| {
        if link::ignored_for_lto(crate_info, cnum) {
            return
        }
        each_linked_rlib_for_lto.push((cnum, path.to_path_buf()));
    }));

    let cgcx = CodegenContext {
        crate_types: sess.crate_types.borrow().clone(),
        each_linked_rlib_for_lto,
        lto: sess.lto(),
        no_landing_pads: sess.no_landing_pads(),
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

        // This flag tracks whether all items have gone through translations
        let mut translation_done = false;

        // This is the queue of LLVM work items that still need processing.
        let mut work_items = Vec::new();

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
                        maybe_start_llvm_timer(&item, &mut llvm_start_time);
                        main_thread_worker_state = MainThreadWorkerState::LLVMing;
                        spawn_work(cgcx, item);
                    }
                }
            } else {
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
                            maybe_start_llvm_timer(&item, &mut llvm_start_time);
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

                maybe_start_llvm_timer(&item, &mut llvm_start_time);

                let cgcx = CodegenContext {
                    worker: get_worker_id(&mut free_worker_ids),
                    .. cgcx.clone()
                };

                spawn_work(cgcx, item);
                running += 1;
            }

            // Relinquish accidentally acquired extra tokens
            tokens.truncate(running);

            match coordinator_receive.recv().unwrap() {
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

                Message::TranslationDone { llvm_work_item, cost, is_last } => {
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

                    if is_last {
                        // If this is the last, don't request a token because
                        // the trans worker thread will be free to handle this
                        // immediately.
                        translation_done = true;
                    } else {
                        helper.request_token();
                    }

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
                Message::Done { result: Err(()), worker_id: _ } => {
                    shared_emitter.fatal("aborting due to worker thread panic");
                    // Exit the coordinator thread
                    panic!("aborting due to worker thread panic")
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

        let compiled_metadata_module = compiled_metadata_module
            .expect("Metadata module not compiled?");

        CompiledModules {
            modules: compiled_modules,
            metadata_module: compiled_metadata_module,
            allocator_module: compiled_allocator_module,
        }
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

    fn maybe_start_llvm_timer(work_item: &WorkItem,
                              llvm_start_time: &mut Option<Instant>) {
        // We keep track of the -Ztime-passes output manually,
        // since the closure-based interface does not fit well here.
        if work_item.config.time_passes {
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
            coordinator_send: Sender<Message>,
            result: Option<CompiledModule>,
            worker_id: usize,
        }
        impl Drop for Bomb {
            fn drop(&mut self) {
                let result = match self.result.take() {
                    Some(compiled_module) => Ok(compiled_module),
                    None => Err(())
                };

                drop(self.coordinator_send.send(Message::Done {
                    result,
                    worker_id: self.worker_id,
                }));
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
        // Note that we ignore the result coming out of `execute_work_item`
        // which will tell us if the worker failed with a `FatalError`. If that
        // has happened, however, then a diagnostic was sent off to the main
        // thread, along with an `AbortIfErrors` message. In that case the main
        // thread is already exiting anyway most likely.
        //
        // In any case, there's no need for us to take further action here, so
        // we just ignore the result and then send off our message saying that
        // we're done, which if `execute_work_item` failed is unlikely to be
        // seen by the main thread, but hey we might as well try anyway.
        bomb.result = {
            let _timing_guard = cgcx.time_graph
                                .as_ref()
                                .map(|tg| tg.start(time_graph::TimelineId(cgcx.worker),
                                                   LLVM_WORK_PACKAGE_KIND));
            Some(execute_work_item(&cgcx, work).unwrap())
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
                                         pname,
                                         prog.status))
                    .note(&format!("{:?}", &cmd))
                    .note(str::from_utf8(&note[..]).unwrap())
                    .emit();
                sess.abort_if_errors();
            }
        },
        Err(e) => {
            sess.err(&format!("could not exec the linker `{}`: {}", pname, e));
            sess.abort_if_errors();
        }
    }
}

pub unsafe fn with_llvm_pmb(llmod: ModuleRef,
                            config: &ModuleConfig,
                            f: &mut FnMut(llvm::PassManagerBuilderRef)) {
    // Create the PassManagerBuilder for LLVM. We configure it with
    // reasonable defaults and prepare it to actually populate the pass
    // manager.
    let builder = llvm::LLVMPassManagerBuilderCreate();
    let opt_level = config.opt_level.unwrap_or(llvm::CodeGenOptLevel::None);
    let opt_size = config.opt_size.unwrap_or(llvm::CodeGenOptSizeNone);
    let inline_threshold = config.inline_threshold;

    llvm::LLVMRustConfigurePassManagerBuilder(builder, opt_level,
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
                                                   &code,
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

    output_filenames: OutputFilenames,
    regular_module_config: ModuleConfig,
    metadata_module_config: ModuleConfig,
    allocator_module_config: ModuleConfig,

    time_graph: Option<TimeGraph>,
    coordinator_send: Sender<Message>,
    trans_worker_receive: Receiver<Message>,
    shared_emitter_main: SharedEmitterMain,
    future: thread::JoinHandle<CompiledModules>,
}

impl OngoingCrateTranslation {
    pub fn join(self, sess: &Session) -> CrateTranslation {
        self.shared_emitter_main.check(sess, true);
        let compiled_modules = match self.future.join() {
            Ok(compiled_modules) => compiled_modules,
            Err(_) => {
                sess.fatal("Error during translation/LLVM phase.");
            }
        };

        sess.abort_if_errors();

        if let Some(time_graph) = self.time_graph {
            time_graph.dump(&format!("{}-timings", self.crate_name));
        }

        copy_module_artifacts_into_incr_comp_cache(sess,
                                                   &compiled_modules,
                                                   &self.output_filenames);
        produce_final_output_artifacts(sess,
                                       &compiled_modules,
                                       &self.output_filenames);

        // FIXME: time_llvm_passes support - does this use a global context or
        // something?
        if sess.opts.cg.codegen_units == 1 && sess.time_llvm_passes() {
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

    pub fn submit_translated_module_to_llvm(&self,
                                            sess: &Session,
                                            mtrans: ModuleTranslation,
                                            cost: u64,
                                            is_last: bool) {
        let module_config = match mtrans.kind {
            ModuleKind::Regular => self.regular_module_config.clone(sess),
            ModuleKind::Metadata => self.metadata_module_config.clone(sess),
            ModuleKind::Allocator => self.allocator_module_config.clone(sess),
        };

        let llvm_work_item = build_work_item(mtrans,
                                             module_config,
                                             self.output_filenames.clone());

        drop(self.coordinator_send.send(Message::TranslationDone {
            llvm_work_item,
            cost,
            is_last
        }));
    }

    pub fn submit_pre_translated_module_to_llvm(&self,
                                                sess: &Session,
                                                mtrans: ModuleTranslation,
                                                is_last: bool) {
        self.wait_for_signal_to_translate_item();
        self.check_for_errors(sess);

        // These are generally cheap and won't through off scheduling.
        let cost = 0;
        self.submit_translated_module_to_llvm(sess, mtrans, cost, is_last);
    }

    pub fn check_for_errors(&self, sess: &Session) {
        self.shared_emitter_main.check(sess, false);
    }

    pub fn wait_for_signal_to_translate_item(&self) {
        match self.trans_worker_receive.recv() {
            Ok(Message::TranslateItem) => {
                // Nothing to do
            }
            Ok(message) => {
                panic!("unexpected message: {:?}", message)
            }
            Err(_) => {
                // One of the LLVM threads must have panicked, fall through so
                // error handling can be reached.
            }
        }
    }
}
