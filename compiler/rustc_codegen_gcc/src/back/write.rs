use std::{env, fs};

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

        let module_name = module.name.clone();
        let module_name = Some(&module_name[..]);

        let _bc_out = cgcx.output_filenames.temp_path(OutputType::Bitcode, module_name);
        let obj_out = cgcx.output_filenames.temp_path(OutputType::Object, module_name);

        if config.bitcode_needed() {
            // TODO(antoyo)
        }

        if config.emit_ir {
            unimplemented!();
        }

        if config.emit_asm {
            let _timer = cgcx
                .prof
                .generic_activity_with_arg("LLVM_module_codegen_emit_asm", &*module.name);
            let path = cgcx.output_filenames.temp_path(OutputType::Assembly, module_name);
            context.compile_to_file(OutputKind::Assembler, path.to_str().expect("path to str"));
        }

        match config.emit_obj {
            EmitObj::ObjectCode(_) => {
                let _timer = cgcx
                    .prof
                    .generic_activity_with_arg("LLVM_module_codegen_emit_obj", &*module.name);
                if env::var("CG_GCCJIT_DUMP_MODULE_NAMES").as_deref() == Ok("1") {
                    println!("Module {}", module.name);
                }
                if env::var("CG_GCCJIT_DUMP_MODULE").as_deref() == Ok(&module.name) {
                    println!("Dumping reproducer {}", module.name);
                    let _ = fs::create_dir("/tmp/reproducers");
                    // FIXME(antoyo): segfault in dump_reproducer_to_file() might be caused by
                    // transmuting an rvalue to an lvalue.
                    // Segfault is actually in gcc::jit::reproducer::get_identifier_as_lvalue
                    context.dump_reproducer_to_file(&format!("/tmp/reproducers/{}.c", module.name));
                    println!("Dumped reproducer {}", module.name);
                }
                context.compile_to_file(OutputKind::ObjectFile, obj_out.to_str().expect("path to str"));
            }

            EmitObj::Bitcode => {
                // TODO(antoyo)
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
