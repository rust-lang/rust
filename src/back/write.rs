use std::{env, fs};

use gccjit::OutputKind;
use rustc_codegen_ssa::{CompiledModule, ModuleCodegen};
use rustc_codegen_ssa::back::link::ensure_removed;
use rustc_codegen_ssa::back::write::{BitcodeSection, CodegenContext, EmitObj, ModuleConfig};
use rustc_errors::Handler;
use rustc_fs_util::link_or_copy;
use rustc_session::config::OutputType;
use rustc_span::fatal_error::FatalError;
use rustc_target::spec::SplitDebuginfo;

use crate::{GccCodegenBackend, GccContext};
use crate::errors::CopyBitcode;

pub(crate) unsafe fn codegen(cgcx: &CodegenContext<GccCodegenBackend>, diag_handler: &Handler, module: ModuleCodegen<GccContext>, config: &ModuleConfig) -> Result<CompiledModule, FatalError> {
    let _timer = cgcx.prof.generic_activity_with_arg("GCC_module_codegen", &*module.name);
    {
        let context = &module.module_llvm.context;

        let module_name = module.name.clone();

        let should_combine_object_files = module.module_llvm.should_combine_object_files;

        let module_name = Some(&module_name[..]);

        // NOTE: Only generate object files with GIMPLE when this environment variable is set for
        // now because this requires a particular setup (same gcc/lto1/lto-wrapper commit as libgccjit).
        let fat_lto = env::var("EMBED_LTO_BITCODE").as_deref() == Ok("1");

        let bc_out = cgcx.output_filenames.temp_path(OutputType::Bitcode, module_name);
        let obj_out = cgcx.output_filenames.temp_path(OutputType::Object, module_name);

        if config.bitcode_needed() && fat_lto {
            let _timer = cgcx
                .prof
                .generic_activity_with_arg("GCC_module_codegen_make_bitcode", &*module.name);

            // TODO(antoyo)
            /*if let Some(bitcode_filename) = bc_out.file_name() {
                cgcx.prof.artifact_size(
                    "llvm_bitcode",
                    bitcode_filename.to_string_lossy(),
                    data.len() as u64,
                );
            }*/

            if config.emit_bc || config.emit_obj == EmitObj::Bitcode {
                let _timer = cgcx
                    .prof
                    .generic_activity_with_arg("GCC_module_codegen_emit_bitcode", &*module.name);
                context.add_command_line_option("-flto=auto");
                context.add_command_line_option("-flto-partition=one");
                context.compile_to_file(OutputKind::ObjectFile, bc_out.to_str().expect("path to str"));
            }

            if config.emit_obj == EmitObj::ObjectCode(BitcodeSection::Full) {
                let _timer = cgcx
                    .prof
                    .generic_activity_with_arg("GCC_module_codegen_embed_bitcode", &*module.name);
                // TODO(antoyo): maybe we should call embed_bitcode to have the proper iOS fixes?
                //embed_bitcode(cgcx, llcx, llmod, &config.bc_cmdline, data);

                context.add_command_line_option("-flto=auto");
                context.add_command_line_option("-flto-partition=one");
                context.add_command_line_option("-ffat-lto-objects");
                // TODO(antoyo): Send -plugin/usr/lib/gcc/x86_64-pc-linux-gnu/11.1.0/liblto_plugin.so to linker (this should be done when specifying the appropriate rustc cli argument).
                context.compile_to_file(OutputKind::ObjectFile, bc_out.to_str().expect("path to str"));
            }
        }

        if config.emit_ir {
            unimplemented!();
        }

        if config.emit_asm {
            let _timer = cgcx
                .prof
                .generic_activity_with_arg("GCC_module_codegen_emit_asm", &*module.name);
            let path = cgcx.output_filenames.temp_path(OutputType::Assembly, module_name);
            context.compile_to_file(OutputKind::Assembler, path.to_str().expect("path to str"));
        }

        match config.emit_obj {
            EmitObj::ObjectCode(_) => {
                let _timer = cgcx
                    .prof
                    .generic_activity_with_arg("GCC_module_codegen_emit_obj", &*module.name);
                if env::var("CG_GCCJIT_DUMP_MODULE_NAMES").as_deref() == Ok("1") {
                    println!("Module {}", module.name);
                }
                if env::var("CG_GCCJIT_DUMP_ALL_MODULES").as_deref() == Ok("1") || env::var("CG_GCCJIT_DUMP_MODULE").as_deref() == Ok(&module.name) {
                    println!("Dumping reproducer {}", module.name);
                    let _ = fs::create_dir("/tmp/reproducers");
                    // FIXME(antoyo): segfault in dump_reproducer_to_file() might be caused by
                    // transmuting an rvalue to an lvalue.
                    // Segfault is actually in gcc::jit::reproducer::get_identifier_as_lvalue
                    context.dump_reproducer_to_file(&format!("/tmp/reproducers/{}.c", module.name));
                    println!("Dumped reproducer {}", module.name);
                }
                if env::var("CG_GCCJIT_DUMP_TO_FILE").as_deref() == Ok("1") {
                    let _ = fs::create_dir("/tmp/gccjit_dumps");
                    let path = &format!("/tmp/gccjit_dumps/{}.c", module.name);
                    context.set_debug_info(true);
                    context.dump_to_file(path, true);
                }
                if should_combine_object_files && fat_lto {
                    context.add_command_line_option("-flto=auto");
                    context.add_command_line_option("-flto-partition=one");

                    context.add_driver_option("-Wl,-r");
                    // NOTE: we need -nostdlib, otherwise, we get the following error:
                    // /usr/bin/ld: cannot find -lgcc_s: No such file or directory
                    context.add_driver_option("-nostdlib");
                    // NOTE: without -fuse-linker-plugin, we get the following error:
                    // lto1: internal compiler error: decompressed stream: Destination buffer is too small
                    context.add_driver_option("-fuse-linker-plugin");

                    // NOTE: this doesn't actually generate an executable. With the above flags, it combines the .o files together in another .o.
                    context.compile_to_file(OutputKind::Executable, obj_out.to_str().expect("path to str"));
                }
                else {
                    context.compile_to_file(OutputKind::ObjectFile, obj_out.to_str().expect("path to str"));
                }
            }

            EmitObj::Bitcode => {
                debug!("copying bitcode {:?} to obj {:?}", bc_out, obj_out);
                if let Err(err) = link_or_copy(&bc_out, &obj_out) {
                    diag_handler.emit_err(CopyBitcode { err });
                }

                if !config.emit_bc {
                    debug!("removing_bitcode {:?}", bc_out);
                    ensure_removed(diag_handler, &bc_out);
                }
            }

            EmitObj::None => {}
        }
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
}

pub(crate) fn save_temp_bitcode(cgcx: &CodegenContext<GccCodegenBackend>, _module: &ModuleCodegen<GccContext>, _name: &str) {
    if !cgcx.save_temps {
        return;
    }
    unimplemented!();
    /*unsafe {
        let ext = format!("{}.bc", name);
        let cgu = Some(&module.name[..]);
        let path = cgcx.output_filenames.temp_path_ext(&ext, cgu);
        let cstr = path_to_c_string(&path);
        let llmod = module.module_llvm.llmod();
        llvm::LLVMWriteBitcodeToFile(llmod, cstr.as_ptr());
    }*/
}
