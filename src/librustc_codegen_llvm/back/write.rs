use crate::attributes;
use crate::back::bytecode;
use crate::back::lto::ThinBuffer;
use crate::base;
use crate::consts;
use crate::llvm::{self, DiagnosticInfo, PassManager, SMDiagnostic};
use crate::llvm_util;
use crate::ModuleLlvm;
use crate::type_::Type;
use crate::context::{is_pie_binary, get_reloc_model};
use crate::common;
use crate::LlvmCodegenBackend;
use rustc::hir::def_id::LOCAL_CRATE;
use rustc_codegen_ssa::back::write::{CodegenContext, ModuleConfig, run_assembler};
use rustc_codegen_ssa::traits::*;
use rustc::session::config::{self, OutputType, Passes, Lto, SwitchWithOptPath};
use rustc::session::Session;
use rustc::ty::TyCtxt;
use rustc_codegen_ssa::{RLIB_BYTECODE_EXTENSION, ModuleCodegen, CompiledModule};
use rustc::util::common::time_ext;
use rustc_fs_util::{path_to_c_string, link_or_copy};
use rustc_data_structures::small_c_str::SmallCStr;
use errors::{Handler, FatalError};

use std::ffi::{CString, CStr};
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::str;
use std::sync::Arc;
use std::slice;
use libc::{c_uint, c_void, c_char, size_t};

pub const RELOC_MODEL_ARGS : [(&str, llvm::RelocMode); 7] = [
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

pub const TLS_MODEL_ARGS : [(&str, llvm::ThreadLocalMode); 4] = [
    ("global-dynamic", llvm::ThreadLocalMode::GeneralDynamic),
    ("local-dynamic", llvm::ThreadLocalMode::LocalDynamic),
    ("initial-exec", llvm::ThreadLocalMode::InitialExec),
    ("local-exec", llvm::ThreadLocalMode::LocalExec),
];

pub fn llvm_err(handler: &errors::Handler, msg: &str) -> FatalError {
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
        let output_c = path_to_c_string(output);
        let result = llvm::LLVMRustWriteOutputFile(target, pm, m, output_c.as_ptr(), file_type);
        result.into_result().map_err(|()| {
            let msg = format!("could not write output to {}", output.display());
            llvm_err(handler, &msg)
        })
    }
}

pub fn create_informational_target_machine(
    sess: &Session,
    find_features: bool,
) -> &'static mut llvm::TargetMachine {
    target_machine_factory(sess, config::OptLevel::No, find_features)().unwrap_or_else(|err| {
        llvm_err(sess.diagnostic(), &err).raise()
    })
}

pub fn create_target_machine(
    tcx: TyCtxt<'_>,
    find_features: bool,
) -> &'static mut llvm::TargetMachine {
    target_machine_factory(&tcx.sess, tcx.backend_optimization_level(LOCAL_CRATE), find_features)()
    .unwrap_or_else(|err| {
        llvm_err(tcx.sess.diagnostic(), &err).raise()
    })
}

pub fn to_llvm_opt_settings(cfg: config::OptLevel) -> (llvm::CodeGenOptLevel, llvm::CodeGenOptSize)
{
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

// If find_features is true this won't access `sess.crate_types` by assuming
// that `is_pie_binary` is false. When we discover LLVM target features
// `sess.crate_types` is uninitialized so we cannot access it.
pub fn target_machine_factory(sess: &Session, optlvl: config::OptLevel, find_features: bool)
    -> Arc<dyn Fn() -> Result<&'static mut llvm::TargetMachine, String> + Send + Sync>
{
    let reloc_model = get_reloc_model(sess);

    let (opt_level, _) = to_llvm_opt_settings(optlvl);
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

pub(crate) fn save_temp_bitcode(
    cgcx: &CodegenContext<LlvmCodegenBackend>,
    module: &ModuleCodegen<ModuleLlvm>,
    name: &str
) {
    if !cgcx.save_temps {
        return
    }
    unsafe {
        let ext = format!("{}.bc", name);
        let cgu = Some(&module.name[..]);
        let path = cgcx.output_filenames.temp_path_ext(&ext, cgu);
        let cstr = path_to_c_string(&path);
        let llmod = module.module_llvm.llmod();
        llvm::LLVMWriteBitcodeToFile(llmod, cstr.as_ptr());
    }
}

pub struct DiagnosticHandlers<'a> {
    data: *mut (&'a CodegenContext<LlvmCodegenBackend>, &'a Handler),
    llcx: &'a llvm::Context,
}

impl<'a> DiagnosticHandlers<'a> {
    pub fn new(cgcx: &'a CodegenContext<LlvmCodegenBackend>,
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

unsafe extern "C" fn report_inline_asm(cgcx: &CodegenContext<LlvmCodegenBackend>,
                                       msg: &str,
                                       cookie: c_uint) {
    cgcx.diag_emitter.inline_asm_error(cookie as u32, msg.to_owned());
}

unsafe extern "C" fn inline_asm_handler(diag: &SMDiagnostic,
                                        user: *const c_void,
                                        cookie: c_uint) {
    if user.is_null() {
        return
    }
    let (cgcx, _) = *(user as *const (&CodegenContext<LlvmCodegenBackend>, &Handler));

    let msg = llvm::build_string(|s| llvm::LLVMRustWriteSMDiagnosticToString(diag, s))
        .expect("non-UTF8 SMDiagnostic");

    report_inline_asm(cgcx, &msg, cookie);
}

unsafe extern "C" fn diagnostic_handler(info: &DiagnosticInfo, user: *mut c_void) {
    if user.is_null() {
        return
    }
    let (cgcx, diag_handler) = *(user as *const (&CodegenContext<LlvmCodegenBackend>, &Handler));

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
pub(crate) unsafe fn optimize(cgcx: &CodegenContext<LlvmCodegenBackend>,
                   diag_handler: &Handler,
                   module: &ModuleCodegen<ModuleLlvm>,
                   config: &ModuleConfig)
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
        let out = path_to_c_string(&out);
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
            let using_thin_buffers = config.bitcode_needed();
            let mut have_name_anon_globals_pass = false;
            if !config.no_prepopulate_passes {
                llvm::LLVMRustAddAnalysisPasses(tm, fpm, llmod);
                llvm::LLVMRustAddAnalysisPasses(tm, mpm, llmod);
                let opt_level = config.opt_level.map(|x| to_llvm_opt_settings(x).0)
                    .unwrap_or(llvm::CodeGenOptLevel::None);
                let prepare_for_thin_lto = cgcx.lto == Lto::Thin || cgcx.lto == Lto::ThinLocal ||
                    (cgcx.lto != Lto::Fat && cgcx.opts.cg.linker_plugin_lto.enabled());
                with_llvm_pmb(llmod, &config, opt_level, prepare_for_thin_lto, &mut |b| {
                    llvm::LLVMPassManagerBuilderPopulateFunctionPassManager(b, fpm);
                    llvm::LLVMPassManagerBuilderPopulateModulePassManager(b, mpm);
                });

                have_name_anon_globals_pass = have_name_anon_globals_pass || prepare_for_thin_lto;
                if using_thin_buffers && !prepare_for_thin_lto {
                    assert!(addpass("name-anon-globals"));
                    have_name_anon_globals_pass = true;
                }
            }

            for pass in &config.passes {
                if !addpass(pass) {
                    diag_handler.warn(&format!("unknown pass `{}`, ignoring", pass));
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
        {
            let _timer = cgcx.profile_activity("LLVM_function_passes");
            time_ext(config.time_passes,
                        None,
                        &format!("llvm function passes [{}]", module_name.unwrap()),
                        || {
                llvm::LLVMRustRunFunctionPassManager(fpm, llmod)
            });
        }
        {
            let _timer = cgcx.profile_activity("LLVM_module_passes");
            time_ext(config.time_passes,
                    None,
                    &format!("llvm module passes [{}]", module_name.unwrap()),
                    || {
                llvm::LLVMRunPassManager(mpm, llmod)
            });
        }

        // Deallocate managers that we're now done with
        llvm::LLVMDisposePassManager(fpm);
        llvm::LLVMDisposePassManager(mpm);
    }
    Ok(())
}

pub(crate) unsafe fn codegen(cgcx: &CodegenContext<LlvmCodegenBackend>,
                  diag_handler: &Handler,
                  module: ModuleCodegen<ModuleLlvm>,
                  config: &ModuleConfig)
    -> Result<CompiledModule, FatalError>
{
    let _timer = cgcx.profile_activity("codegen");
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
            let _timer = cgcx.profile_activity("LLVM_make_bitcode");
            let thin = ThinBuffer::new(llmod);
            let data = thin.data();

            if write_bc {
                let _timer = cgcx.profile_activity("LLVM_emit_bitcode");
                if let Err(e) = fs::write(&bc_out, data) {
                    let msg = format!("failed to write bytecode to {}: {}", bc_out.display(), e);
                    diag_handler.err(&msg);
                }
            }

            if config.embed_bitcode {
                let _timer = cgcx.profile_activity("LLVM_embed_bitcode");
                embed_bitcode(cgcx, llcx, llmod, Some(data));
            }

            if config.emit_bc_compressed {
                let _timer = cgcx.profile_activity("LLVM_compress_bitcode");
                let dst = bc_out.with_extension(RLIB_BYTECODE_EXTENSION);
                let data = bytecode::encode(&module.name, data);
                if let Err(e) = fs::write(&dst, data) {
                    let msg = format!("failed to write bytecode to {}: {}", dst.display(), e);
                    diag_handler.err(&msg);
                }
            }
        } else if config.embed_bitcode_marker {
            embed_bitcode(cgcx, llcx, llmod, None);
        }

        time_ext(config.time_passes, None, &format!("codegen passes [{}]", module_name.unwrap()),
            || -> Result<(), FatalError> {
            if config.emit_ir {
                let _timer = cgcx.profile_activity("LLVM_emit_ir");
                let out = cgcx.output_filenames.temp_path(OutputType::LlvmAssembly, module_name);
                let out_c = path_to_c_string(&out);

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
                    let result =
                        llvm::LLVMRustPrintModule(cpm, llmod, out_c.as_ptr(), demangle_callback);
                    llvm::LLVMDisposePassManager(cpm);
                    result.into_result().map_err(|()| {
                        let msg = format!("failed to write LLVM IR to {}", out.display());
                        llvm_err(diag_handler, &msg)
                    })
                })?;
            }

            if config.emit_asm || asm_to_obj {
                let _timer = cgcx.profile_activity("LLVM_emit_asm");
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
            }

            if write_obj {
                let _timer = cgcx.profile_activity("LLVM_emit_obj");
                with_codegen(tm, llmod, config.no_builtins, |cpm| {
                    write_output_file(diag_handler, tm, cpm, llmod, &obj_out,
                                      llvm::FileType::ObjectFile)
                })?;
            } else if asm_to_obj {
                let _timer = cgcx.profile_activity("LLVM_asm_to_obj");
                let assembly = cgcx.output_filenames.temp_path(OutputType::Assembly, module_name);
                run_assembler(cgcx, diag_handler, &assembly, &obj_out);

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
unsafe fn embed_bitcode(cgcx: &CodegenContext<LlvmCodegenBackend>,
                        llcx: &llvm::Context,
                        llmod: &llvm::Module,
                        bitcode: Option<&[u8]>) {
    let llconst = common::bytes_in_context(llcx, bitcode.unwrap_or(&[]));
    let llglobal = llvm::LLVMAddGlobal(
        llmod,
        common::val_ty(llconst),
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

    let llconst = common::bytes_in_context(llcx, &[]);
    let llglobal = llvm::LLVMAddGlobal(
        llmod,
        common::val_ty(llconst),
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
    let opt_size = config.opt_size.map(|x| to_llvm_opt_settings(x).1)
        .unwrap_or(llvm::CodeGenOptSizeNone);
    let inline_threshold = config.inline_threshold;

    let pgo_gen_path = match config.pgo_gen {
        SwitchWithOptPath::Enabled(ref opt_dir_path) => {
            let path = if let Some(dir_path) = opt_dir_path {
                dir_path.join("default_%m.profraw")
            } else {
                PathBuf::from("default_%m.profraw")
            };

            Some(CString::new(format!("{}", path.display())).unwrap())
        }
        SwitchWithOptPath::Disabled => {
            None
        }
    };

    let pgo_use_path = config.pgo_use.as_ref().map(|path_buf| {
        CString::new(path_buf.to_string_lossy().as_bytes()).unwrap()
    });

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

// Create a `__imp_<symbol> = &symbol` global for every public static `symbol`.
// This is required to satisfy `dllimport` references to static data in .rlibs
// when using MSVC linker.  We do this only for data, as linker can fix up
// code references on its own.
// See #26591, #27438
fn create_msvc_imps(
    cgcx: &CodegenContext<LlvmCodegenBackend>,
    llcx: &llvm::Context,
    llmod: &llvm::Module
) {
    if !cgcx.msvc_imps_needed {
        return
    }
    // The x86 ABI seems to require that leading underscores are added to symbol
    // names, so we need an extra underscore on x86. There's also a leading
    // '\x01' here which disables LLVM's symbol mangling (e.g., no extra
    // underscores added in front).
    let prefix = if cgcx.target_arch == "x86" {
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
            .filter_map(|val| {
                // Exclude some symbols that we know are not Rust symbols.
                let name = CStr::from_ptr(llvm::LLVMGetValueName(val));
                if ignored(name.to_bytes()) {
                    None
                } else {
                    Some((val, name))
                }
            })
            .map(move |(val, name)| {
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

    // Use this function to exclude certain symbols from `__imp` generation.
    fn ignored(symbol_name: &[u8]) -> bool {
        // These are symbols generated by LLVM's profiling instrumentation
        symbol_name.starts_with(b"__llvm_profile_")
    }
}
