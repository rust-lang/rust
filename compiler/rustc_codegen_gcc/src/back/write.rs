use std::{env, fs};

use gccjit::{Context, OutputKind};
use rustc_codegen_ssa::back::link::ensure_removed;
use rustc_codegen_ssa::back::write::{BitcodeSection, CodegenContext, EmitObj, ModuleConfig};
use rustc_codegen_ssa::{CompiledModule, ModuleCodegen};
use rustc_errors::DiagCtxtHandle;
use rustc_fs_util::link_or_copy;
use rustc_session::config::OutputType;
use rustc_span::fatal_error::FatalError;
use rustc_target::spec::SplitDebuginfo;

use crate::base::add_pic_option;
use crate::errors::CopyBitcode;
use crate::{GccCodegenBackend, GccContext};

pub(crate) fn codegen(
    cgcx: &CodegenContext<GccCodegenBackend>,
    dcx: DiagCtxtHandle<'_>,
    module: ModuleCodegen<GccContext>,
    config: &ModuleConfig,
) -> Result<CompiledModule, FatalError> {
    let _timer = cgcx.prof.generic_activity_with_arg("GCC_module_codegen", &*module.name);
    {
        let context = &module.module_llvm.context;

        let should_combine_object_files = module.module_llvm.should_combine_object_files;

        // NOTE: Only generate object files with GIMPLE when this environment variable is set for
        // now because this requires a particular setup (same gcc/lto1/lto-wrapper commit as libgccjit).
        // TODO(antoyo): remove this environment variable.
        let fat_lto = env::var("EMBED_LTO_BITCODE").as_deref() == Ok("1");

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
            if fat_lto {
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
                    let _timer = cgcx.prof.generic_activity_with_arg(
                        "GCC_module_codegen_emit_bitcode",
                        &*module.name,
                    );
                    context.add_command_line_option("-flto=auto");
                    context.add_command_line_option("-flto-partition=one");
                    // TODO(antoyo): remove since we don't want fat objects when it is for Bitcode only.
                    context.add_command_line_option("-ffat-lto-objects");
                    context.compile_to_file(
                        OutputKind::ObjectFile,
                        bc_out.to_str().expect("path to str"),
                    );
                }

                if config.emit_obj == EmitObj::ObjectCode(BitcodeSection::Full) {
                    let _timer = cgcx.prof.generic_activity_with_arg(
                        "GCC_module_codegen_embed_bitcode",
                        &*module.name,
                    );
                    // TODO(antoyo): maybe we should call embed_bitcode to have the proper iOS fixes?
                    //embed_bitcode(cgcx, llcx, llmod, &config.bc_cmdline, data);

                    context.add_command_line_option("-flto=auto");
                    context.add_command_line_option("-flto-partition=one");
                    context.add_command_line_option("-ffat-lto-objects");
                    // TODO(antoyo): Send -plugin/usr/lib/gcc/x86_64-pc-linux-gnu/11.1.0/liblto_plugin.so to linker (this should be done when specifying the appropriate rustc cli argument).
                    context.compile_to_file(
                        OutputKind::ObjectFile,
                        bc_out.to_str().expect("path to str"),
                    );
                }
            } else {
                if config.emit_bc || config.emit_obj == EmitObj::Bitcode {
                    let _timer = cgcx.prof.generic_activity_with_arg(
                        "GCC_module_codegen_emit_bitcode",
                        &*module.name,
                    );
                    context.compile_to_file(
                        OutputKind::ObjectFile,
                        bc_out.to_str().expect("path to str"),
                    );
                }

                if config.emit_obj == EmitObj::ObjectCode(BitcodeSection::Full) {
                    // TODO(antoyo): we might want to emit to emit an error here, saying to set the
                    // environment variable EMBED_LTO_BITCODE.
                    let _timer = cgcx.prof.generic_activity_with_arg(
                        "GCC_module_codegen_embed_bitcode",
                        &*module.name,
                    );
                    // TODO(antoyo): maybe we should call embed_bitcode to have the proper iOS fixes?
                    //embed_bitcode(cgcx, llcx, llmod, &config.bc_cmdline, data);

                    // TODO(antoyo): Send -plugin/usr/lib/gcc/x86_64-pc-linux-gnu/11.1.0/liblto_plugin.so to linker (this should be done when specifying the appropriate rustc cli argument).
                    context.compile_to_file(
                        OutputKind::ObjectFile,
                        bc_out.to_str().expect("path to str"),
                    );
                }
            }
        }

        if config.emit_ir {
            let out = cgcx.output_filenames.temp_path_for_cgu(
                OutputType::LlvmAssembly,
                &module.name,
                cgcx.invocation_temp.as_deref(),
            );
            std::fs::write(out, "").expect("write file");
        }

        if config.emit_asm {
            let _timer =
                cgcx.prof.generic_activity_with_arg("GCC_module_codegen_emit_asm", &*module.name);
            let path = cgcx.output_filenames.temp_path_for_cgu(
                OutputType::Assembly,
                &module.name,
                cgcx.invocation_temp.as_deref(),
            );
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
                if env::var("CG_GCCJIT_DUMP_ALL_MODULES").as_deref() == Ok("1")
                    || env::var("CG_GCCJIT_DUMP_MODULE").as_deref() == Ok(&module.name)
                {
                    println!("Dumping reproducer {}", module.name);
                    let _ = fs::create_dir("/tmp/reproducers");
                    // FIXME(antoyo): segfault in dump_reproducer_to_file() might be caused by
                    // transmuting an rvalue to an lvalue.
                    // Segfault is actually in gcc::jit::reproducer::get_identifier_as_lvalue
                    context.dump_reproducer_to_file(format!("/tmp/reproducers/{}.c", module.name));
                    println!("Dumped reproducer {}", module.name);
                }
                if env::var("CG_GCCJIT_DUMP_TO_FILE").as_deref() == Ok("1") {
                    let _ = fs::create_dir("/tmp/gccjit_dumps");
                    let path = &format!("/tmp/gccjit_dumps/{}.c", module.name);
                    context.set_debug_info(true);
                    context.dump_to_file(path, true);
                }
                if should_combine_object_files {
                    if fat_lto {
                        context.add_command_line_option("-flto=auto");
                        context.add_command_line_option("-flto-partition=one");

                        // NOTE: without -fuse-linker-plugin, we get the following error:
                        // lto1: internal compiler error: decompressed stream: Destination buffer is too small
                        // TODO(antoyo): since we do not do LTO when the linker is invoked anymore, perhaps
                        // the following flag is not necessary anymore.
                        context.add_driver_option("-fuse-linker-plugin");
                    }

                    context.add_driver_option("-Wl,-r");
                    // NOTE: we need -nostdlib, otherwise, we get the following error:
                    // /usr/bin/ld: cannot find -lgcc_s: No such file or directory
                    context.add_driver_option("-nostdlib");

                    let path = obj_out.to_str().expect("path to str");

                    if fat_lto {
                        let lto_path = format!("{}.lto", path);
                        // cSpell:disable
                        // FIXME(antoyo): The LTO frontend generates the following warning:
                        // ../build_sysroot/sysroot_src/library/core/src/num/dec2flt/lemire.rs:150:15: warning: type of ‘_ZN4core3num7dec2flt5table17POWER_OF_FIVE_12817ha449a68fb31379e4E’ does not match original declaration [-Wlto-type-mismatch]
                        // 150 |     let (lo5, hi5) = POWER_OF_FIVE_128[index];
                        //     |               ^
                        // lto1: note: ‘_ZN4core3num7dec2flt5table17POWER_OF_FIVE_12817ha449a68fb31379e4E’ was previously declared here
                        //
                        // This option is to mute it to make the UI tests pass with LTO enabled.
                        // cSpell:enable
                        context.add_driver_option("-Wno-lto-type-mismatch");
                        // NOTE: this doesn't actually generate an executable. With the above
                        // flags, it combines the .o files together in another .o.
                        context.compile_to_file(OutputKind::Executable, &lto_path);

                        let context = Context::default();
                        if cgcx.target_arch == "x86" || cgcx.target_arch == "x86_64" {
                            // NOTE: it seems we need to use add_driver_option instead of
                            // add_command_line_option here because we use the LTO frontend via gcc.
                            context.add_driver_option("-masm=intel");
                        }

                        // NOTE: these two options are needed to invoke LTO to produce an object file.
                        // We need to initiate a second compilation because the arguments "-x lto"
                        // needs to be at the very beginning.
                        context.add_driver_option("-x");
                        context.add_driver_option("lto");
                        add_pic_option(&context, module.module_llvm.relocation_model);
                        context.add_driver_option(lto_path);

                        context.compile_to_file(OutputKind::ObjectFile, path);
                    } else {
                        // NOTE: this doesn't actually generate an executable. With the above
                        // flags, it combines the .o files together in another .o.
                        context.compile_to_file(OutputKind::Executable, path);
                    }
                } else {
                    context.compile_to_file(
                        OutputKind::ObjectFile,
                        obj_out.to_str().expect("path to str"),
                    );
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
    }

    Ok(module.into_compiled_module(
        config.emit_obj != EmitObj::None,
        cgcx.target_can_use_split_dwarf && cgcx.split_debuginfo == SplitDebuginfo::Unpacked,
        config.emit_bc,
        config.emit_asm,
        config.emit_ir,
        &cgcx.output_filenames,
        cgcx.invocation_temp.as_deref(),
    ))
}

pub(crate) fn link(
    _cgcx: &CodegenContext<GccCodegenBackend>,
    _dcx: DiagCtxtHandle<'_>,
    mut _modules: Vec<ModuleCodegen<GccContext>>,
) -> Result<ModuleCodegen<GccContext>, FatalError> {
    unimplemented!();
}

pub(crate) fn save_temp_bitcode(
    cgcx: &CodegenContext<GccCodegenBackend>,
    _module: &ModuleCodegen<GccContext>,
    _name: &str,
) {
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
