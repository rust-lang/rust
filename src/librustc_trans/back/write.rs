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
use back::link::{get_linker, remove};
use session::config::{OutputFilenames, Passes, SomePasses, AllPasses};
use session::Session;
use session::config::{self, OutputType};
use llvm;
use llvm::{ModuleRef, TargetMachineRef, PassManagerRef, DiagnosticInfoRef, ContextRef};
use llvm::SMDiagnosticRef;
use trans::{CrateTranslation, ModuleTranslation};
use util::common::time;
use util::common::path2cstr;
use syntax::codemap;
use syntax::errors::{self, Handler, Level};
use syntax::errors::emitter::Emitter;

use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::fs;
use std::path::{Path, PathBuf};
use std::ptr;
use std::str;
use std::sync::{Arc, Mutex};
use std::sync::mpsc::channel;
use std::thread;
use libc::{self, c_uint, c_int, c_void};

pub fn llvm_err(handler: &errors::Handler, msg: String) -> ! {
    unsafe {
        let cstr = llvm::LLVMRustGetLastError();
        if cstr == ptr::null() {
            panic!(handler.fatal(&msg[..]));
        } else {
            let err = CStr::from_ptr(cstr).to_bytes();
            let err = String::from_utf8_lossy(err).to_string();
            libc::free(cstr as *mut _);
            panic!(handler.fatal(&format!("{}: {}", &msg[..], &err[..])));
        }
    }
}

pub fn write_output_file(
        handler: &errors::Handler,
        target: llvm::TargetMachineRef,
        pm: llvm::PassManagerRef,
        m: ModuleRef,
        output: &Path,
        file_type: llvm::FileType) {
    unsafe {
        let output_c = path2cstr(output);
        let result = llvm::LLVMRustWriteOutputFile(
                target, pm, m, output_c.as_ptr(), file_type);
        if !result {
            llvm_err(handler, format!("could not write output to {}", output.display()));
        }
    }
}


struct Diagnostic {
    msg: String,
    code: Option<String>,
    lvl: Level,
}

// We use an Arc instead of just returning a list of diagnostics from the
// child thread because we need to make sure that the messages are seen even
// if the child thread panics (for example, when `fatal` is called).
#[derive(Clone)]
struct SharedEmitter {
    buffer: Arc<Mutex<Vec<Diagnostic>>>,
}

impl SharedEmitter {
    fn new() -> SharedEmitter {
        SharedEmitter {
            buffer: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn dump(&mut self, handler: &Handler) {
        let mut buffer = self.buffer.lock().unwrap();
        for diag in &*buffer {
            match diag.code {
                Some(ref code) => {
                    handler.emit_with_code(None,
                                           &diag.msg,
                                           &code[..],
                                           diag.lvl);
                },
                None => {
                    handler.emit(None,
                                 &diag.msg,
                                 diag.lvl);
                },
            }
        }
        buffer.clear();
    }
}

impl Emitter for SharedEmitter {
    fn emit(&mut self, sp: Option<codemap::Span>,
            msg: &str, code: Option<&str>, lvl: Level) {
        assert!(sp.is_none(), "SharedEmitter doesn't support spans");

        self.buffer.lock().unwrap().push(Diagnostic {
            msg: msg.to_string(),
            code: code.map(|s| s.to_string()),
            lvl: lvl,
        });
    }

    fn custom_emit(&mut self, _sp: errors::RenderSpan, _msg: &str, _lvl: Level) {
        panic!("SharedEmitter doesn't support custom_emit");
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
    format!("{},{}", sess.target.target.options.features, sess.opts.cg.target_feature)
}

fn get_llvm_opt_level(optimize: config::OptLevel) -> llvm::CodeGenOptLevel {
    match optimize {
      config::No => llvm::CodeGenLevelNone,
      config::Less => llvm::CodeGenLevelLess,
      config::Default => llvm::CodeGenLevelDefault,
      config::Aggressive => llvm::CodeGenLevelAggressive,
    }
}

pub fn create_target_machine(sess: &Session) -> TargetMachineRef {
    let reloc_model_arg = match sess.opts.cg.relocation_model {
        Some(ref s) => &s[..],
        None => &sess.target.target.options.relocation_model[..],
    };
    let reloc_model = match reloc_model_arg {
        "pic" => llvm::RelocPIC,
        "static" => llvm::RelocStatic,
        "default" => llvm::RelocDefault,
        "dynamic-no-pic" => llvm::RelocDynamicNoPic,
        _ => {
            sess.err(&format!("{:?} is not a valid relocation mode",
                             sess.opts
                                 .cg
                                 .relocation_model));
            sess.abort_if_errors();
            unreachable!();
        }
    };

    let opt_level = get_llvm_opt_level(sess.opts.optimize);
    let use_softfp = sess.opts.cg.soft_float;

    let any_library = sess.crate_types.borrow().iter().any(|ty| {
        *ty != config::CrateTypeExecutable
    });

    let ffunction_sections = sess.target.target.options.function_sections;
    let fdata_sections = ffunction_sections;

    let code_model_arg = match sess.opts.cg.code_model {
        Some(ref s) => &s[..],
        None => &sess.target.target.options.code_model[..],
    };

    let code_model = match code_model_arg {
        "default" => llvm::CodeModelDefault,
        "small" => llvm::CodeModelSmall,
        "kernel" => llvm::CodeModelKernel,
        "medium" => llvm::CodeModelMedium,
        "large" => llvm::CodeModelLarge,
        _ => {
            sess.err(&format!("{:?} is not a valid code model",
                             sess.opts
                                 .cg
                                 .code_model));
            sess.abort_if_errors();
            unreachable!();
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
            !any_library && reloc_model == llvm::RelocPIC,
            ffunction_sections,
            fdata_sections,
        )
    };

    if tm.is_null() {
        llvm_err(sess.diagnostic(),
                 format!("Could not create LLVM TargetMachine for triple: {}",
                         triple).to_string());
    } else {
        return tm;
    };
}


/// Module-specific configuration for `optimize_and_codegen`.
#[derive(Clone)]
pub struct ModuleConfig {
    /// LLVM TargetMachine to use for codegen.
    tm: TargetMachineRef,
    /// Names of additional optimization passes to run.
    passes: Vec<String>,
    /// Some(level) to optimize at a certain level, or None to run
    /// absolutely no optimizations (used for the metadata module).
    opt_level: Option<llvm::CodeGenOptLevel>,

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
    fn new(tm: TargetMachineRef, passes: Vec<String>) -> ModuleConfig {
        ModuleConfig {
            tm: tm,
            passes: passes,
            opt_level: None,

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

    fn set_flags(&mut self, sess: &Session, trans: &CrateTranslation) {
        self.no_verify = sess.no_verify();
        self.no_prepopulate_passes = sess.opts.cg.no_prepopulate_passes;
        self.no_builtins = trans.no_builtins;
        self.time_passes = sess.time_passes();
        self.inline_threshold = sess.opts.cg.inline_threshold;
        self.obj_is_bitcode = sess.target.target.options.obj_is_bitcode;

        // Copy what clang does by turning on loop vectorization at O2 and
        // slp vectorization at O3. Otherwise configure other optimization aspects
        // of this pass manager builder.
        self.vectorize_loop = !sess.opts.cg.no_vectorize_loops &&
                             (sess.opts.optimize == config::Default ||
                              sess.opts.optimize == config::Aggressive);
        self.vectorize_slp = !sess.opts.cg.no_vectorize_slp &&
                            sess.opts.optimize == config::Aggressive;

        self.merge_functions = sess.opts.optimize == config::Default ||
                               sess.opts.optimize == config::Aggressive;
    }
}

/// Additional resources used by optimize_and_codegen (not module specific)
struct CodegenContext<'a> {
    // Extra resources used for LTO: (sess, reachable).  This will be `None`
    // when running in a worker thread.
    lto_ctxt: Option<(&'a Session, &'a [String])>,
    // Handler to use for diagnostics produced during codegen.
    handler: &'a Handler,
    // LLVM passes added by plugins.
    plugin_passes: Vec<String>,
    // LLVM optimizations for which we want to print remarks.
    remark: Passes,
    // Worker thread number
    worker: usize,
}

impl<'a> CodegenContext<'a> {
    fn new_with_session(sess: &'a Session, reachable: &'a [String]) -> CodegenContext<'a> {
        CodegenContext {
            lto_ctxt: Some((sess, reachable)),
            handler: sess.diagnostic(),
            plugin_passes: sess.plugin_llvm_passes.borrow().clone(),
            remark: sess.opts.cg.remark.clone(),
            worker: 0,
        }
    }
}

struct HandlerFreeVars<'a> {
    llcx: ContextRef,
    cgcx: &'a CodegenContext<'a>,
}

unsafe extern "C" fn report_inline_asm<'a, 'b>(cgcx: &'a CodegenContext<'a>,
                                               msg: &'b str,
                                               cookie: c_uint) {
    use syntax::codemap::ExpnId;

    match cgcx.lto_ctxt {
        Some((sess, _)) => {
            sess.codemap().with_expn_info(ExpnId::from_u32(cookie), |info| match info {
                Some(ei) => sess.span_err(ei.call_site, msg),
                None     => sess.err(msg),
            });
        }

        None => {
            cgcx.handler.err(msg);
            cgcx.handler.note("build without -C codegen-units for more exact errors");
        }
    }
}

unsafe extern "C" fn inline_asm_handler(diag: SMDiagnosticRef,
                                        user: *const c_void,
                                        cookie: c_uint) {
    let HandlerFreeVars { cgcx, .. } = *(user as *const HandlerFreeVars);

    let msg = llvm::build_string(|s| llvm::LLVMWriteSMDiagnosticToString(diag, s))
        .expect("non-UTF8 SMDiagnostic");

    report_inline_asm(cgcx, &msg[..], cookie);
}

unsafe extern "C" fn diagnostic_handler(info: DiagnosticInfoRef, user: *mut c_void) {
    let HandlerFreeVars { llcx, cgcx } = *(user as *const HandlerFreeVars);

    match llvm::diagnostic::Diagnostic::unpack(info) {
        llvm::diagnostic::InlineAsm(inline) => {
            report_inline_asm(cgcx,
                              &*llvm::twine_to_string(inline.message),
                              inline.cookie);
        }

        llvm::diagnostic::Optimization(opt) => {
            let pass_name = str::from_utf8(CStr::from_ptr(opt.pass_name).to_bytes())
                                .ok()
                                .expect("got a non-UTF8 pass name from LLVM");
            let enabled = match cgcx.remark {
                AllPasses => true,
                SomePasses(ref v) => v.iter().any(|s| *s == pass_name),
            };

            if enabled {
                let loc = llvm::debug_loc_to_string(llcx, opt.debug_loc);
                cgcx.handler.note(&format!("optimization {} for {} at {}: {}",
                                           opt.kind.describe(),
                                           pass_name,
                                           if loc.is_empty() { "[unknown]" } else { &*loc },
                                           llvm::twine_to_string(opt.message)));
            }
        }

        _ => (),
    }
}

// Unsafe due to LLVM calls.
unsafe fn optimize_and_codegen(cgcx: &CodegenContext,
                               mtrans: ModuleTranslation,
                               config: ModuleConfig,
                               name_extra: String,
                               output_names: OutputFilenames) {
    let ModuleTranslation { llmod, llcx } = mtrans;
    let tm = config.tm;

    // llcx doesn't outlive this function, so we can put this on the stack.
    let fv = HandlerFreeVars {
        llcx: llcx,
        cgcx: cgcx,
    };
    let fv = &fv as *const HandlerFreeVars as *mut c_void;

    llvm::LLVMSetInlineAsmDiagnosticHandler(llcx, inline_asm_handler, fv);
    llvm::LLVMContextSetDiagnosticHandler(llcx, diagnostic_handler, fv);

    if config.emit_no_opt_bc {
        let ext = format!("{}.no-opt.bc", name_extra);
        let out = output_names.with_extension(&ext);
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
        let addpass = |pass: &str| {
            let pass = CString::new(pass).unwrap();
            llvm::LLVMRustAddPass(fpm, pass.as_ptr())
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
                cgcx.handler.warn(&format!("unknown pass `{}`, ignoring",
                                           pass));
            }
        }

        for pass in &cgcx.plugin_passes {
            if !addpass(pass) {
                cgcx.handler.err(&format!("a plugin asked for LLVM pass \
                                           `{}` but LLVM does not \
                                           recognize it", pass));
            }
        }

        cgcx.handler.abort_if_errors();

        // Finally, run the actual optimization passes
        time(config.time_passes, &format!("llvm function passes [{}]", cgcx.worker), ||
             llvm::LLVMRustRunFunctionPassManager(fpm, llmod));
        time(config.time_passes, &format!("llvm module passes [{}]", cgcx.worker), ||
             llvm::LLVMRunPassManager(mpm, llmod));

        // Deallocate managers that we're now done with
        llvm::LLVMDisposePassManager(fpm);
        llvm::LLVMDisposePassManager(mpm);

        match cgcx.lto_ctxt {
            Some((sess, reachable)) if sess.lto() =>  {
                time(sess.time_passes(), "all lto passes", ||
                     lto::run(sess, llmod, tm, reachable, &config,
                              &name_extra, &output_names));

                if config.emit_lto_bc {
                    let name = format!("{}.lto.bc", name_extra);
                    let out = output_names.with_extension(&name);
                    let out = path2cstr(&out);
                    llvm::LLVMWriteBitcodeToFile(llmod, out.as_ptr());
                }
            },
            _ => {},
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
    unsafe fn with_codegen<F>(tm: TargetMachineRef,
                              llmod: ModuleRef,
                              no_builtins: bool,
                              f: F) where
        F: FnOnce(PassManagerRef),
    {
        let cpm = llvm::LLVMCreatePassManager();
        llvm::LLVMRustAddAnalysisPasses(tm, cpm, llmod);
        llvm::LLVMRustAddLibraryInfo(cpm, llmod, no_builtins);
        f(cpm);
    }

    // Change what we write and cleanup based on whether obj files are
    // just llvm bitcode. In that case write bitcode, and possibly
    // delete the bitcode if it wasn't requisted. Don't generate the
    // machine code, instead copy the .o file from the .bc
    let write_bc = config.emit_bc || config.obj_is_bitcode;
    let rm_bc = !config.emit_bc && config.obj_is_bitcode;
    let write_obj = config.emit_obj && !config.obj_is_bitcode;
    let copy_bc_to_obj = config.emit_obj && config.obj_is_bitcode;

    let bc_out = output_names.with_extension(&format!("{}.bc", name_extra));
    let obj_out = output_names.with_extension(&format!("{}.o", name_extra));

    if write_bc {
        let bc_out_c = path2cstr(&bc_out);
        llvm::LLVMWriteBitcodeToFile(llmod, bc_out_c.as_ptr());
    }

    time(config.time_passes, &format!("codegen passes [{}]", cgcx.worker), || {
        if config.emit_ir {
            let ext = format!("{}.ll", name_extra);
            let out = output_names.with_extension(&ext);
            let out = path2cstr(&out);
            with_codegen(tm, llmod, config.no_builtins, |cpm| {
                llvm::LLVMRustPrintModule(cpm, llmod, out.as_ptr());
                llvm::LLVMDisposePassManager(cpm);
            })
        }

        if config.emit_asm {
            let path = output_names.with_extension(&format!("{}.s", name_extra));

            // We can't use the same module for asm and binary output, because that triggers
            // various errors like invalid IR or broken binaries, so we might have to clone the
            // module to produce the asm output
            let llmod = if config.emit_obj {
                llvm::LLVMCloneModule(llmod)
            } else {
                llmod
            };
            with_codegen(tm, llmod, config.no_builtins, |cpm| {
                write_output_file(cgcx.handler, tm, cpm, llmod, &path,
                                  llvm::AssemblyFileType);
            });
            if config.emit_obj {
                llvm::LLVMDisposeModule(llmod);
            }
        }

        if write_obj {
            with_codegen(tm, llmod, config.no_builtins, |cpm| {
                write_output_file(cgcx.handler, tm, cpm, llmod, &obj_out, llvm::ObjectFileType);
            });
        }
    });

    if copy_bc_to_obj {
        debug!("copying bitcode {:?} to obj {:?}", bc_out, obj_out);
        if let Err(e) = fs::copy(&bc_out, &obj_out) {
            cgcx.handler.err(&format!("failed to copy bitcode to object file: {}", e));
        }
    }

    if rm_bc {
        debug!("removing_bitcode {:?}", bc_out);
        if let Err(e) = fs::remove_file(&bc_out) {
            cgcx.handler.err(&format!("failed to remove bitcode: {}", e));
        }
    }

    llvm::LLVMDisposeModule(llmod);
    llvm::LLVMContextDispose(llcx);
    llvm::LLVMRustDisposeTargetMachine(tm);
}

pub fn run_passes(sess: &Session,
                  trans: &CrateTranslation,
                  output_types: &HashMap<OutputType, Option<PathBuf>>,
                  crate_output: &OutputFilenames) {
    // It's possible that we have `codegen_units > 1` but only one item in
    // `trans.modules`.  We could theoretically proceed and do LTO in that
    // case, but it would be confusing to have the validity of
    // `-Z lto -C codegen-units=2` depend on details of the crate being
    // compiled, so we complain regardless.
    if sess.lto() && sess.opts.cg.codegen_units > 1 {
        // This case is impossible to handle because LTO expects to be able
        // to combine the entire crate and all its dependencies into a
        // single compilation unit, but each codegen unit is in a separate
        // LLVM context, so they can't easily be combined.
        sess.fatal("can't perform LTO when using multiple codegen units");
    }

    // Sanity check
    assert!(trans.modules.len() == sess.opts.cg.codegen_units);

    let tm = create_target_machine(sess);

    // Figure out what we actually need to build.

    let mut modules_config = ModuleConfig::new(tm, sess.opts.cg.passes.clone());
    let mut metadata_config = ModuleConfig::new(tm, vec!());

    modules_config.opt_level = Some(get_llvm_opt_level(sess.opts.optimize));

    // Save all versions of the bytecode if we're saving our temporaries.
    if sess.opts.cg.save_temps {
        modules_config.emit_no_opt_bc = true;
        modules_config.emit_bc = true;
        modules_config.emit_lto_bc = true;
        metadata_config.emit_bc = true;
    }

    // Emit bitcode files for the crate if we're emitting an rlib.
    // Whenever an rlib is created, the bitcode is inserted into the
    // archive in order to allow LTO against it.
    let needs_crate_bitcode =
            sess.crate_types.borrow().contains(&config::CrateTypeRlib) &&
            sess.opts.output_types.contains_key(&OutputType::Exe);
    let needs_crate_object =
            sess.opts.output_types.contains_key(&OutputType::Exe);
    if needs_crate_bitcode {
        modules_config.emit_bc = true;
    }

    for output_type in output_types.keys() {
        match *output_type {
            OutputType::Bitcode => { modules_config.emit_bc = true; },
            OutputType::LlvmAssembly => { modules_config.emit_ir = true; },
            OutputType::Assembly => {
                modules_config.emit_asm = true;
                // If we're not using the LLVM assembler, this function
                // could be invoked specially with output_type_assembly, so
                // in this case we still want the metadata object file.
                if !sess.opts.output_types.contains_key(&OutputType::Assembly) {
                    metadata_config.emit_obj = true;
                }
            },
            OutputType::Object => { modules_config.emit_obj = true; },
            OutputType::Exe => {
                modules_config.emit_obj = true;
                metadata_config.emit_obj = true;
            },
            OutputType::DepInfo => {}
        }
    }

    modules_config.set_flags(sess, trans);
    metadata_config.set_flags(sess, trans);


    // Populate a buffer with a list of codegen threads.  Items are processed in
    // LIFO order, just because it's a tiny bit simpler that way.  (The order
    // doesn't actually matter.)
    let mut work_items = Vec::with_capacity(1 + trans.modules.len());

    {
        let work = build_work_item(sess,
                                   trans.metadata_module,
                                   metadata_config.clone(),
                                   crate_output.clone(),
                                   "metadata".to_string());
        work_items.push(work);
    }

    for (index, mtrans) in trans.modules.iter().enumerate() {
        let work = build_work_item(sess,
                                   *mtrans,
                                   modules_config.clone(),
                                   crate_output.clone(),
                                   format!("{}", index));
        work_items.push(work);
    }

    // Process the work items, optionally using worker threads.
    if sess.opts.cg.codegen_units == 1 {
        run_work_singlethreaded(sess, &trans.reachable, work_items);
    } else {
        run_work_multithreaded(sess, work_items, sess.opts.cg.codegen_units);
    }

    // All codegen is finished.
    unsafe {
        llvm::LLVMRustDisposeTargetMachine(tm);
    }

    // Produce final compile outputs.
    let copy_gracefully = |from: &Path, to: &Path| {
        if let Err(e) = fs::copy(from, to) {
            sess.err(&format!("could not copy {:?} to {:?}: {}", from, to, e));
        }
    };

    let copy_if_one_unit = |ext: &str,
                            output_type: OutputType,
                            keep_numbered: bool| {
        if sess.opts.cg.codegen_units == 1 {
            // 1) Only one codegen unit.  In this case it's no difficulty
            //    to copy `foo.0.x` to `foo.x`.
            copy_gracefully(&crate_output.with_extension(ext),
                            &crate_output.path(output_type));
            if !sess.opts.cg.save_temps && !keep_numbered {
                // The user just wants `foo.x`, not `foo.0.x`.
                remove(sess, &crate_output.with_extension(ext));
            }
        } else if crate_output.outputs.contains_key(&output_type) {
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
    };

    // Flag to indicate whether the user explicitly requested bitcode.
    // Otherwise, we produced it only as a temporary output, and will need
    // to get rid of it.
    let mut user_wants_bitcode = false;
    let mut user_wants_objects = false;
    for output_type in output_types.keys() {
        match *output_type {
            OutputType::Bitcode => {
                user_wants_bitcode = true;
                // Copy to .bc, but always keep the .0.bc.  There is a later
                // check to figure out if we should delete .0.bc files, or keep
                // them for making an rlib.
                copy_if_one_unit("0.bc", OutputType::Bitcode, true);
            }
            OutputType::LlvmAssembly => {
                copy_if_one_unit("0.ll", OutputType::LlvmAssembly, false);
            }
            OutputType::Assembly => {
                copy_if_one_unit("0.s", OutputType::Assembly, false);
            }
            OutputType::Object => {
                user_wants_objects = true;
                copy_if_one_unit("0.o", OutputType::Object, true);
            }
            OutputType::Exe |
            OutputType::DepInfo => {}
        }
    }
    let user_wants_bitcode = user_wants_bitcode;

    // Clean up unwanted temporary files.

    // We create the following files by default:
    //  - crate.0.bc
    //  - crate.0.o
    //  - crate.metadata.bc
    //  - crate.metadata.o
    //  - crate.o (linked from crate.##.o)
    //  - crate.bc (copied from crate.0.bc)
    // We may create additional files if requested by the user (through
    // `-C save-temps` or `--emit=` flags).

    if !sess.opts.cg.save_temps {
        // Remove the temporary .0.o objects.  If the user didn't
        // explicitly request bitcode (with --emit=bc), and the bitcode is not
        // needed for building an rlib, then we must remove .0.bc as well.

        // Specific rules for keeping .0.bc:
        //  - If we're building an rlib (`needs_crate_bitcode`), then keep
        //    it.
        //  - If the user requested bitcode (`user_wants_bitcode`), and
        //    codegen_units > 1, then keep it.
        //  - If the user requested bitcode but codegen_units == 1, then we
        //    can toss .0.bc because we copied it to .bc earlier.
        //  - If we're not building an rlib and the user didn't request
        //    bitcode, then delete .0.bc.
        // If you change how this works, also update back::link::link_rlib,
        // where .0.bc files are (maybe) deleted after making an rlib.
        let keep_numbered_bitcode = needs_crate_bitcode ||
                (user_wants_bitcode && sess.opts.cg.codegen_units > 1);

        let keep_numbered_objects = needs_crate_object ||
                (user_wants_objects && sess.opts.cg.codegen_units > 1);

        for i in 0..trans.modules.len() {
            if modules_config.emit_obj && !keep_numbered_objects {
                let ext = format!("{}.o", i);
                remove(sess, &crate_output.with_extension(&ext));
            }

            if modules_config.emit_bc && !keep_numbered_bitcode {
                let ext = format!("{}.bc", i);
                remove(sess, &crate_output.with_extension(&ext));
            }
        }

        if metadata_config.emit_bc && !user_wants_bitcode {
            remove(sess, &crate_output.with_extension("metadata.bc"));
        }
    }

    // We leave the following files around by default:
    //  - crate.o
    //  - crate.metadata.o
    //  - crate.bc
    // These are used in linking steps and will be cleaned up afterward.

    // FIXME: time_llvm_passes support - does this use a global context or
    // something?
    if sess.opts.cg.codegen_units == 1 && sess.time_llvm_passes() {
        unsafe { llvm::LLVMRustPrintPassTimings(); }
    }
}

struct WorkItem {
    mtrans: ModuleTranslation,
    config: ModuleConfig,
    output_names: OutputFilenames,
    name_extra: String
}

fn build_work_item(sess: &Session,
                   mtrans: ModuleTranslation,
                   config: ModuleConfig,
                   output_names: OutputFilenames,
                   name_extra: String)
                   -> WorkItem
{
    let mut config = config;
    config.tm = create_target_machine(sess);
    WorkItem { mtrans: mtrans, config: config, output_names: output_names,
               name_extra: name_extra }
}

fn execute_work_item(cgcx: &CodegenContext,
                     work_item: WorkItem) {
    unsafe {
        optimize_and_codegen(cgcx, work_item.mtrans, work_item.config,
                             work_item.name_extra, work_item.output_names);
    }
}

fn run_work_singlethreaded(sess: &Session,
                           reachable: &[String],
                           work_items: Vec<WorkItem>) {
    let cgcx = CodegenContext::new_with_session(sess, reachable);

    // Since we're running single-threaded, we can pass the session to
    // the proc, allowing `optimize_and_codegen` to perform LTO.
    for work in work_items.into_iter().rev() {
        execute_work_item(&cgcx, work);
    }
}

fn run_work_multithreaded(sess: &Session,
                          work_items: Vec<WorkItem>,
                          num_workers: usize) {
    // Run some workers to process the work items.
    let work_items_arc = Arc::new(Mutex::new(work_items));
    let mut diag_emitter = SharedEmitter::new();
    let mut futures = Vec::with_capacity(num_workers);

    for i in 0..num_workers {
        let work_items_arc = work_items_arc.clone();
        let diag_emitter = diag_emitter.clone();
        let plugin_passes = sess.plugin_llvm_passes.borrow().clone();
        let remark = sess.opts.cg.remark.clone();

        let (tx, rx) = channel();
        let mut tx = Some(tx);
        futures.push(rx);

        thread::Builder::new().name(format!("codegen-{}", i)).spawn(move || {
            let diag_handler = Handler::with_emitter(true, false, box diag_emitter);

            // Must construct cgcx inside the proc because it has non-Send
            // fields.
            let cgcx = CodegenContext {
                lto_ctxt: None,
                handler: &diag_handler,
                plugin_passes: plugin_passes,
                remark: remark,
                worker: i,
            };

            loop {
                // Avoid holding the lock for the entire duration of the match.
                let maybe_work = work_items_arc.lock().unwrap().pop();
                match maybe_work {
                    Some(work) => {
                        execute_work_item(&cgcx, work);

                        // Make sure to fail the worker so the main thread can
                        // tell that there were errors.
                        cgcx.handler.abort_if_errors();
                    }
                    None => break,
                }
            }

            tx.take().unwrap().send(()).unwrap();
        }).unwrap();
    }

    let mut panicked = false;
    for rx in futures {
        match rx.recv() {
            Ok(()) => {},
            Err(_) => {
                panicked = true;
            },
        }
        // Display any new diagnostics.
        diag_emitter.dump(sess.diagnostic());
    }
    if panicked {
        sess.fatal("aborting due to worker thread panic");
    }
}

pub fn run_assembler(sess: &Session, outputs: &OutputFilenames) {
    let (pname, mut cmd) = get_linker(sess);

    cmd.arg("-c").arg("-o").arg(&outputs.path(OutputType::Object))
                           .arg(&outputs.temp_path(OutputType::Assembly));
    debug!("{:?}", cmd);

    match cmd.output() {
        Ok(prog) => {
            if !prog.status.success() {
                sess.err(&format!("linking with `{}` failed: {}",
                                 pname,
                                 prog.status));
                sess.note(&format!("{:?}", &cmd));
                let mut note = prog.stderr.clone();
                note.extend_from_slice(&prog.stdout);
                sess.note(str::from_utf8(&note[..]).unwrap());
                sess.abort_if_errors();
            }
        },
        Err(e) => {
            sess.err(&format!("could not exec the linker `{}`: {}", pname, e));
            sess.abort_if_errors();
        }
    }
}

pub unsafe fn configure_llvm(sess: &Session) {
    let mut llvm_c_strs = Vec::new();
    let mut llvm_args = Vec::new();

    {
        let mut add = |arg: &str| {
            let s = CString::new(arg).unwrap();
            llvm_args.push(s.as_ptr());
            llvm_c_strs.push(s);
        };
        add("rustc"); // fake program name
        if sess.time_llvm_passes() { add("-time-passes"); }
        if sess.print_llvm_passes() { add("-debug-pass=Structure"); }

        // FIXME #21627 disable faulty FastISel on AArch64 (even for -O0)
        if sess.target.target.arch == "aarch64" { add("-fast-isel=0"); }

        for arg in &sess.opts.cg.llvm_args {
            add(&(*arg));
        }
    }

    llvm::LLVMInitializePasses();

    llvm::initialize_available_targets();

    llvm::LLVMRustSetLLVMOptions(llvm_args.len() as c_int,
                                 llvm_args.as_ptr());
}

pub unsafe fn with_llvm_pmb(llmod: ModuleRef,
                            config: &ModuleConfig,
                            f: &mut FnMut(llvm::PassManagerBuilderRef)) {
    // Create the PassManagerBuilder for LLVM. We configure it with
    // reasonable defaults and prepare it to actually populate the pass
    // manager.
    let builder = llvm::LLVMPassManagerBuilderCreate();
    let opt = config.opt_level.unwrap_or(llvm::CodeGenLevelNone);
    let inline_threshold = config.inline_threshold;

    llvm::LLVMRustConfigurePassManagerBuilder(builder, opt,
                                              config.merge_functions,
                                              config.vectorize_slp,
                                              config.vectorize_loop);

    llvm::LLVMRustAddBuilderLibraryInfo(builder, llmod, config.no_builtins);

    // Here we match what clang does (kinda). For O0 we only inline
    // always-inline functions (but don't add lifetime intrinsics), at O1 we
    // inline with lifetime intrinsics, and O2+ we add an inliner with a
    // thresholds copied from clang.
    match (opt, inline_threshold) {
        (_, Some(t)) => {
            llvm::LLVMPassManagerBuilderUseInlinerWithThreshold(builder, t as u32);
        }
        (llvm::CodeGenLevelNone, _) => {
            llvm::LLVMRustAddAlwaysInlinePass(builder, false);
        }
        (llvm::CodeGenLevelLess, _) => {
            llvm::LLVMRustAddAlwaysInlinePass(builder, true);
        }
        (llvm::CodeGenLevelDefault, _) => {
            llvm::LLVMPassManagerBuilderUseInlinerWithThreshold(builder, 225);
        }
        (llvm::CodeGenLevelAggressive, _) => {
            llvm::LLVMPassManagerBuilderUseInlinerWithThreshold(builder, 275);
        }
    }

    f(builder);
    llvm::LLVMPassManagerBuilderDispose(builder);
}
