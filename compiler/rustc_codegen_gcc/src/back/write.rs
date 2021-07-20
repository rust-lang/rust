use std::fs;

use gccjit::OutputKind;
use rustc_codegen_ssa::{CompiledModule, ModuleCodegen};
use rustc_codegen_ssa::back::write::{CodegenContext, EmitObj, ModuleConfig};
use rustc_errors::Handler;
use rustc_session::config::OutputType;
use rustc_span::fatal_error::FatalError;
use rustc_target::spec::SplitDebuginfo;

use crate::{GccCodegenBackend, GccContext};

pub(crate) unsafe fn codegen(cgcx: &CodegenContext<GccCodegenBackend>, _diag_handler: &Handler, module: ModuleCodegen<GccContext>, config: &ModuleConfig) -> Result<CompiledModule, FatalError> {
    let _timer = cgcx.prof.generic_activity_with_arg("LLVM_module_codegen", &module.name[..]);
    {
        let context = &module.module_llvm.context;

        //let llcx = &*module.module_llvm.llcx;
        //let tm = &*module.module_llvm.tm;
        let module_name = module.name.clone();
        let module_name = Some(&module_name[..]);
        //let handlers = DiagnosticHandlers::new(cgcx, diag_handler, llcx);

        /*if cgcx.msvc_imps_needed {
            create_msvc_imps(cgcx, llcx, llmod);
        }*/

        // A codegen-specific pass manager is used to generate object
        // files for an GCC module.
        //
        // Apparently each of these pass managers is a one-shot kind of
        // thing, so we create a new one for each type of output. The
        // pass manager passed to the closure should be ensured to not
        // escape the closure itself, and the manager should only be
        // used once.
        /*unsafe fn with_codegen<'ll, F, R>(tm: &'ll llvm::TargetMachine, llmod: &'ll llvm::Module, no_builtins: bool, f: F) -> R
        where F: FnOnce(&'ll mut PassManager<'ll>) -> R,
        {
            let cpm = llvm::LLVMCreatePassManager();
            llvm::LLVMAddAnalysisPasses(tm, cpm);
            llvm::LLVMRustAddLibraryInfo(cpm, llmod, no_builtins);
            f(cpm)
        }*/

        // Two things to note:
        // - If object files are just LLVM bitcode we write bitcode, copy it to
        //   the .o file, and delete the bitcode if it wasn't otherwise
        //   requested.
        // - If we don't have the integrated assembler then we need to emit
        //   asm from LLVM and use `gcc` to create the object file.

        let _bc_out = cgcx.output_filenames.temp_path(OutputType::Bitcode, module_name);
        let obj_out = cgcx.output_filenames.temp_path(OutputType::Object, module_name);

        if config.bitcode_needed() {
            // TODO
            /*let _timer = cgcx
                .prof
                .generic_activity_with_arg("LLVM_module_codegen_make_bitcode", &module.name[..]);
            let thin = ThinBuffer::new(llmod);
            let data = thin.data();

            if config.emit_bc || config.emit_obj == EmitObj::Bitcode {
                let _timer = cgcx.prof.generic_activity_with_arg(
                    "LLVM_module_codegen_emit_bitcode",
                    &module.name[..],
                );
                if let Err(e) = fs::write(&bc_out, data) {
                    let msg = format!("failed to write bytecode to {}: {}", bc_out.display(), e);
                    diag_handler.err(&msg);
                }
            }

            if config.emit_obj == EmitObj::ObjectCode(BitcodeSection::Full) {
                let _timer = cgcx.prof.generic_activity_with_arg(
                    "LLVM_module_codegen_embed_bitcode",
                    &module.name[..],
                );
                embed_bitcode(cgcx, llcx, llmod, Some(data));
            }

            if config.emit_bc_compressed {
                let _timer = cgcx.prof.generic_activity_with_arg(
                    "LLVM_module_codegen_emit_compressed_bitcode",
                    &module.name[..],
                );
                let dst = bc_out.with_extension(RLIB_BYTECODE_EXTENSION);
                let data = bytecode::encode(&module.name, data);
                if let Err(e) = fs::write(&dst, data) {
                    let msg = format!("failed to write bytecode to {}: {}", dst.display(), e);
                    diag_handler.err(&msg);
                }
            }*/
        } /*else if config.emit_obj == EmitObj::ObjectCode(BitcodeSection::Marker) {
            unimplemented!();
            //embed_bitcode(cgcx, llcx, llmod, None);
        }*/

        if config.emit_ir {
            unimplemented!();
            /*let _timer = cgcx
                .prof
                .generic_activity_with_arg("LLVM_module_codegen_emit_ir", &module.name[..]);
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

                if write!(cursor, "{:#}", demangled).is_err() {
                    // Possible only if provided buffer is not big enough
                    return 0;
                }

                cursor.position() as size_t
            }

            let result = llvm::LLVMRustPrintModule(llmod, out_c.as_ptr(), demangle_callback);
            result.into_result().map_err(|()| {
                let msg = format!("failed to write LLVM IR to {}", out.display());
                llvm_err(diag_handler, &msg)
            })?;*/
        }

        if config.emit_asm {
            let _timer = cgcx
                .prof
                .generic_activity_with_arg("LLVM_module_codegen_emit_asm", &module.name[..]);
            let path = cgcx.output_filenames.temp_path(OutputType::Assembly, module_name);
            context.compile_to_file(OutputKind::Assembler, path.to_str().expect("path to str"));

            /*with_codegen(tm, llmod, config.no_builtins, |cpm| {
                write_output_file(diag_handler, tm, cpm, llmod, &path, llvm::FileType::AssemblyFile)
            })?;*/
        }

        match config.emit_obj {
            EmitObj::ObjectCode(_) => {
                let _timer = cgcx
                    .prof
                    .generic_activity_with_arg("LLVM_module_codegen_emit_obj", &module.name[..]);
                //with_codegen(tm, llmod, config.no_builtins, |cpm| {
                    //println!("1: {}", module.name);
                    match &*module.name {
                        "std_example.7rcbfp3g-cgu.15" => {
                            println!("Dumping reproducer {}", module.name);
                            let _ = fs::create_dir("/tmp/reproducers");
                            // FIXME: segfault in dump_reproducer_to_file() might be caused by
                            // transmuting an rvalue to an lvalue.
                            // Segfault is actually in gcc::jit::reproducer::get_identifier_as_lvalue
                            context.dump_reproducer_to_file(&format!("/tmp/reproducers/{}.c", module.name));
                            println!("Dumped reproducer {}", module.name);
                        },
                        _ => (),
                    }
                    /*let _ = fs::create_dir("/tmp/dumps");
                    context.dump_to_file(&format!("/tmp/dumps/{}.c", module.name), true);
                    println!("Dumped {}", module.name);*/
                    //println!("Compile module {}", module.name);
                    context.compile_to_file(OutputKind::ObjectFile, obj_out.to_str().expect("path to str"));
                //})?;
            }

            EmitObj::Bitcode => {
                //unimplemented!();
                /*debug!("copying bitcode {:?} to obj {:?}", bc_out, obj_out);
                if let Err(e) = link_or_copy(&bc_out, &obj_out) {
                    diag_handler.err(&format!("failed to copy bitcode to object file: {}", e));
                }

                if !config.emit_bc {
                    debug!("removing_bitcode {:?}", bc_out);
                    if let Err(e) = fs::remove_file(&bc_out) {
                        diag_handler.err(&format!("failed to remove bitcode: {}", e));
                    }
                }*/
            }

            EmitObj::None => {}
        }

        //drop(handlers);
    }

    Ok(module.into_compiled_module(
        config.emit_obj != EmitObj::None,
        cgcx.target_can_use_split_dwarf && cgcx.split_debuginfo == SplitDebuginfo::Unpacked,
        config.emit_bc,
        &cgcx.output_filenames,
    ))
}

pub(crate) fn link(_cgcx: &CodegenContext<GccCodegenBackend>, _diag_handler: &Handler, mut _modules: Vec<ModuleCodegen<GccContext>>) -> Result<ModuleCodegen<GccContext>, FatalError> {
    unimplemented!();
    /*use super::lto::{Linker, ModuleBuffer};
    // Sort the modules by name to ensure to ensure deterministic behavior.
    modules.sort_by(|a, b| a.name.cmp(&b.name));
    let (first, elements) =
        modules.split_first().expect("Bug! modules must contain at least one module.");

    let mut linker = Linker::new(first.module_llvm.llmod());
    for module in elements {
        let _timer =
            cgcx.prof.generic_activity_with_arg("LLVM_link_module", format!("{:?}", module.name));
        let buffer = ModuleBuffer::new(module.module_llvm.llmod());
        linker.add(&buffer.data()).map_err(|()| {
            let msg = format!("failed to serialize module {:?}", module.name);
            llvm_err(&diag_handler, &msg)
        })?;
    }
    drop(linker);
    Ok(modules.remove(0))*/
}
