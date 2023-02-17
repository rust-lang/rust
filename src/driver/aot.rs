//! The AOT driver uses [`cranelift_object`] to write object files suitable for linking into a
//! standalone executable.

use std::fs::File;
use std::path::PathBuf;
use std::sync::Arc;
use std::thread::JoinHandle;

use rustc_codegen_ssa::back::metadata::create_compressed_metadata_file;
use rustc_codegen_ssa::{CodegenResults, CompiledModule, CrateInfo, ModuleKind};
use rustc_data_structures::profiling::SelfProfilerRef;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_metadata::EncodedMetadata;
use rustc_middle::dep_graph::{WorkProduct, WorkProductId};
use rustc_middle::mir::mono::{CodegenUnit, MonoItem};
use rustc_session::cgu_reuse_tracker::CguReuse;
use rustc_session::config::{DebugInfo, OutputFilenames, OutputType};
use rustc_session::Session;

use cranelift_object::{ObjectBuilder, ObjectModule};

use crate::concurrency_limiter::{ConcurrencyLimiter, ConcurrencyLimiterToken};
use crate::global_asm::GlobalAsmConfig;
use crate::{prelude::*, BackendConfig};

struct ModuleCodegenResult {
    module_regular: CompiledModule,
    module_global_asm: Option<CompiledModule>,
    existing_work_product: Option<(WorkProductId, WorkProduct)>,
}

enum OngoingModuleCodegen {
    Sync(Result<ModuleCodegenResult, String>),
    Async(JoinHandle<Result<ModuleCodegenResult, String>>),
}

impl<HCX> HashStable<HCX> for OngoingModuleCodegen {
    fn hash_stable(&self, _: &mut HCX, _: &mut StableHasher) {
        // do nothing
    }
}

pub(crate) struct OngoingCodegen {
    modules: Vec<OngoingModuleCodegen>,
    allocator_module: Option<CompiledModule>,
    metadata_module: Option<CompiledModule>,
    metadata: EncodedMetadata,
    crate_info: CrateInfo,
    concurrency_limiter: ConcurrencyLimiter,
}

impl OngoingCodegen {
    pub(crate) fn join(
        self,
        sess: &Session,
        backend_config: &BackendConfig,
    ) -> (CodegenResults, FxHashMap<WorkProductId, WorkProduct>) {
        let mut work_products = FxHashMap::default();
        let mut modules = vec![];

        for module_codegen in self.modules {
            let module_codegen_result = match module_codegen {
                OngoingModuleCodegen::Sync(module_codegen_result) => module_codegen_result,
                OngoingModuleCodegen::Async(join_handle) => match join_handle.join() {
                    Ok(module_codegen_result) => module_codegen_result,
                    Err(panic) => std::panic::resume_unwind(panic),
                },
            };

            let module_codegen_result = match module_codegen_result {
                Ok(module_codegen_result) => module_codegen_result,
                Err(err) => sess.fatal(&err),
            };
            let ModuleCodegenResult { module_regular, module_global_asm, existing_work_product } =
                module_codegen_result;

            if let Some((work_product_id, work_product)) = existing_work_product {
                work_products.insert(work_product_id, work_product);
            } else {
                let work_product = if backend_config.disable_incr_cache {
                    None
                } else if let Some(module_global_asm) = &module_global_asm {
                    rustc_incremental::copy_cgu_workproduct_to_incr_comp_cache_dir(
                        sess,
                        &module_regular.name,
                        &[
                            ("o", &module_regular.object.as_ref().unwrap()),
                            ("asm.o", &module_global_asm.object.as_ref().unwrap()),
                        ],
                    )
                } else {
                    rustc_incremental::copy_cgu_workproduct_to_incr_comp_cache_dir(
                        sess,
                        &module_regular.name,
                        &[("o", &module_regular.object.as_ref().unwrap())],
                    )
                };
                if let Some((work_product_id, work_product)) = work_product {
                    work_products.insert(work_product_id, work_product);
                }
            }

            modules.push(module_regular);
            if let Some(module_global_asm) = module_global_asm {
                modules.push(module_global_asm);
            }
        }

        self.concurrency_limiter.finished();

        sess.abort_if_errors();

        (
            CodegenResults {
                modules,
                allocator_module: self.allocator_module,
                metadata_module: self.metadata_module,
                metadata: self.metadata,
                crate_info: self.crate_info,
            },
            work_products,
        )
    }
}

fn make_module(sess: &Session, backend_config: &BackendConfig, name: String) -> ObjectModule {
    let isa = crate::build_isa(sess, backend_config);

    let mut builder =
        ObjectBuilder::new(isa, name + ".o", cranelift_module::default_libcall_names()).unwrap();
    // Unlike cg_llvm, cg_clif defaults to disabling -Zfunction-sections. For cg_llvm binary size
    // is important, while cg_clif cares more about compilation times. Enabling -Zfunction-sections
    // can easily double the amount of time necessary to perform linking.
    builder.per_function_section(sess.opts.unstable_opts.function_sections.unwrap_or(false));
    ObjectModule::new(builder)
}

fn emit_cgu(
    output_filenames: &OutputFilenames,
    prof: &SelfProfilerRef,
    name: String,
    module: ObjectModule,
    debug: Option<DebugContext>,
    unwind_context: UnwindContext,
    global_asm_object_file: Option<PathBuf>,
) -> Result<ModuleCodegenResult, String> {
    let mut product = module.finish();

    if let Some(mut debug) = debug {
        debug.emit(&mut product);
    }

    unwind_context.emit(&mut product);

    let module_regular =
        emit_module(output_filenames, prof, product.object, ModuleKind::Regular, name.clone())?;

    Ok(ModuleCodegenResult {
        module_regular,
        module_global_asm: global_asm_object_file.map(|global_asm_object_file| CompiledModule {
            name: format!("{name}.asm"),
            kind: ModuleKind::Regular,
            object: Some(global_asm_object_file),
            dwarf_object: None,
            bytecode: None,
        }),
        existing_work_product: None,
    })
}

fn emit_module(
    output_filenames: &OutputFilenames,
    prof: &SelfProfilerRef,
    mut object: cranelift_object::object::write::Object<'_>,
    kind: ModuleKind,
    name: String,
) -> Result<CompiledModule, String> {
    if object.format() == cranelift_object::object::BinaryFormat::Elf {
        let comment_section = object.add_section(
            Vec::new(),
            b".comment".to_vec(),
            cranelift_object::object::SectionKind::OtherString,
        );
        let mut producer = vec![0];
        producer.extend(crate::debuginfo::producer().as_bytes());
        producer.push(0);
        object.set_section_data(comment_section, producer, 1);
    }

    let tmp_file = output_filenames.temp_path(OutputType::Object, Some(&name));
    let mut file = match File::create(&tmp_file) {
        Ok(file) => file,
        Err(err) => return Err(format!("error creating object file: {}", err)),
    };

    if let Err(err) = object.write_stream(&mut file) {
        return Err(format!("error writing object file: {}", err));
    }

    prof.artifact_size("object_file", &*name, file.metadata().unwrap().len());

    Ok(CompiledModule { name, kind, object: Some(tmp_file), dwarf_object: None, bytecode: None })
}

fn reuse_workproduct_for_cgu(
    tcx: TyCtxt<'_>,
    cgu: &CodegenUnit<'_>,
) -> Result<ModuleCodegenResult, String> {
    let work_product = cgu.previous_work_product(tcx);
    let obj_out_regular =
        tcx.output_filenames(()).temp_path(OutputType::Object, Some(cgu.name().as_str()));
    let source_file_regular = rustc_incremental::in_incr_comp_dir_sess(
        &tcx.sess,
        &work_product.saved_files.get("o").expect("no saved object file in work product"),
    );

    if let Err(err) = rustc_fs_util::link_or_copy(&source_file_regular, &obj_out_regular) {
        return Err(format!(
            "unable to copy {} to {}: {}",
            source_file_regular.display(),
            obj_out_regular.display(),
            err
        ));
    }
    let obj_out_global_asm =
        crate::global_asm::add_file_stem_postfix(obj_out_regular.clone(), ".asm");
    let has_global_asm = if let Some(asm_o) = work_product.saved_files.get("asm.o") {
        let source_file_global_asm = rustc_incremental::in_incr_comp_dir_sess(&tcx.sess, asm_o);
        if let Err(err) = rustc_fs_util::link_or_copy(&source_file_global_asm, &obj_out_global_asm)
        {
            return Err(format!(
                "unable to copy {} to {}: {}",
                source_file_regular.display(),
                obj_out_regular.display(),
                err
            ));
        }
        true
    } else {
        false
    };

    Ok(ModuleCodegenResult {
        module_regular: CompiledModule {
            name: cgu.name().to_string(),
            kind: ModuleKind::Regular,
            object: Some(obj_out_regular),
            dwarf_object: None,
            bytecode: None,
        },
        module_global_asm: has_global_asm.then(|| CompiledModule {
            name: cgu.name().to_string(),
            kind: ModuleKind::Regular,
            object: Some(obj_out_global_asm),
            dwarf_object: None,
            bytecode: None,
        }),
        existing_work_product: Some((cgu.work_product_id(), work_product)),
    })
}

fn module_codegen(
    tcx: TyCtxt<'_>,
    (backend_config, global_asm_config, cgu_name, token): (
        BackendConfig,
        Arc<GlobalAsmConfig>,
        rustc_span::Symbol,
        ConcurrencyLimiterToken,
    ),
) -> OngoingModuleCodegen {
    let (cgu_name, mut cx, mut module, codegened_functions) =
        tcx.prof.verbose_generic_activity_with_arg("codegen cgu", cgu_name.as_str()).run(|| {
            let cgu = tcx.codegen_unit(cgu_name);
            let mono_items = cgu.items_in_deterministic_order(tcx);

            let mut module = make_module(tcx.sess, &backend_config, cgu_name.as_str().to_string());

            let mut cx = crate::CodegenCx::new(
                tcx,
                backend_config.clone(),
                module.isa(),
                tcx.sess.opts.debuginfo != DebugInfo::None,
                cgu_name,
            );
            super::predefine_mono_items(tcx, &mut module, &mono_items);
            let mut codegened_functions = vec![];
            for (mono_item, _) in mono_items {
                match mono_item {
                    MonoItem::Fn(inst) => {
                        let codegened_function = crate::base::codegen_fn(
                            tcx,
                            &mut cx,
                            Function::new(),
                            &mut module,
                            inst,
                        );
                        codegened_functions.push(codegened_function);
                    }
                    MonoItem::Static(def_id) => {
                        crate::constant::codegen_static(tcx, &mut module, def_id)
                    }
                    MonoItem::GlobalAsm(item_id) => {
                        crate::global_asm::codegen_global_asm_item(
                            tcx,
                            &mut cx.global_asm,
                            item_id,
                        );
                    }
                }
            }
            crate::main_shim::maybe_create_entry_wrapper(
                tcx,
                &mut module,
                &mut cx.unwind_context,
                false,
                cgu.is_primary(),
            );

            let cgu_name = cgu.name().as_str().to_owned();

            (cgu_name, cx, module, codegened_functions)
        });

    OngoingModuleCodegen::Async(std::thread::spawn(move || {
        cx.profiler.clone().verbose_generic_activity_with_arg("compile functions", &*cgu_name).run(
            || {
                let mut cached_context = Context::new();
                for codegened_func in codegened_functions {
                    crate::base::compile_fn(
                        &mut cx,
                        &mut cached_context,
                        &mut module,
                        codegened_func,
                    );
                }
            },
        );

        let global_asm_object_file = cx
            .profiler
            .verbose_generic_activity_with_arg("compile assembly", &*cgu_name)
            .run(|| {
                crate::global_asm::compile_global_asm(&global_asm_config, &cgu_name, &cx.global_asm)
            })?;

        let codegen_result = cx
            .profiler
            .verbose_generic_activity_with_arg("write object file", &*cgu_name)
            .run(|| {
                emit_cgu(
                    &global_asm_config.output_filenames,
                    &cx.profiler,
                    cgu_name,
                    module,
                    cx.debug_context,
                    cx.unwind_context,
                    global_asm_object_file,
                )
            });
        std::mem::drop(token);
        codegen_result
    }))
}

pub(crate) fn run_aot(
    tcx: TyCtxt<'_>,
    backend_config: BackendConfig,
    metadata: EncodedMetadata,
    need_metadata_module: bool,
) -> Box<OngoingCodegen> {
    let cgus = if tcx.sess.opts.output_types.should_codegen() {
        tcx.collect_and_partition_mono_items(()).1
    } else {
        // If only `--emit metadata` is used, we shouldn't perform any codegen.
        // Also `tcx.collect_and_partition_mono_items` may panic in that case.
        &[]
    };

    if tcx.dep_graph.is_fully_enabled() {
        for cgu in cgus {
            tcx.ensure().codegen_unit(cgu.name());
        }
    }

    let global_asm_config = Arc::new(crate::global_asm::GlobalAsmConfig::new(tcx));

    let mut concurrency_limiter = ConcurrencyLimiter::new(tcx.sess, cgus.len());

    let modules = tcx.sess.time("codegen mono items", || {
        cgus.iter()
            .map(|cgu| {
                let cgu_reuse = if backend_config.disable_incr_cache {
                    CguReuse::No
                } else {
                    determine_cgu_reuse(tcx, cgu)
                };
                tcx.sess.cgu_reuse_tracker.set_actual_reuse(cgu.name().as_str(), cgu_reuse);

                match cgu_reuse {
                    CguReuse::No => {
                        let dep_node = cgu.codegen_dep_node(tcx);
                        tcx.dep_graph
                            .with_task(
                                dep_node,
                                tcx,
                                (
                                    backend_config.clone(),
                                    global_asm_config.clone(),
                                    cgu.name(),
                                    concurrency_limiter.acquire(),
                                ),
                                module_codegen,
                                Some(rustc_middle::dep_graph::hash_result),
                            )
                            .0
                    }
                    CguReuse::PreLto => unreachable!(),
                    CguReuse::PostLto => {
                        concurrency_limiter.job_already_done();
                        OngoingModuleCodegen::Sync(reuse_workproduct_for_cgu(tcx, cgu))
                    }
                }
            })
            .collect::<Vec<_>>()
    });

    let mut allocator_module = make_module(tcx.sess, &backend_config, "allocator_shim".to_string());
    let mut allocator_unwind_context = UnwindContext::new(allocator_module.isa(), true);
    let created_alloc_shim =
        crate::allocator::codegen(tcx, &mut allocator_module, &mut allocator_unwind_context);

    let allocator_module = if created_alloc_shim {
        let mut product = allocator_module.finish();
        allocator_unwind_context.emit(&mut product);

        match emit_module(
            tcx.output_filenames(()),
            &tcx.sess.prof,
            product.object,
            ModuleKind::Allocator,
            "allocator_shim".to_owned(),
        ) {
            Ok(allocator_module) => Some(allocator_module),
            Err(err) => tcx.sess.fatal(err),
        }
    } else {
        None
    };

    let metadata_module = if need_metadata_module {
        let (metadata_cgu_name, tmp_file) = tcx.sess.time("write compressed metadata", || {
            use rustc_middle::mir::mono::CodegenUnitNameBuilder;

            let cgu_name_builder = &mut CodegenUnitNameBuilder::new(tcx);
            let metadata_cgu_name = cgu_name_builder
                .build_cgu_name(LOCAL_CRATE, &["crate"], Some("metadata"))
                .as_str()
                .to_string();

            let tmp_file =
                tcx.output_filenames(()).temp_path(OutputType::Metadata, Some(&metadata_cgu_name));

            let symbol_name = rustc_middle::middle::exported_symbols::metadata_symbol_name(tcx);
            let obj = create_compressed_metadata_file(tcx.sess, &metadata, &symbol_name);

            if let Err(err) = std::fs::write(&tmp_file, obj) {
                tcx.sess.fatal(&format!("error writing metadata object file: {}", err));
            }

            (metadata_cgu_name, tmp_file)
        });

        Some(CompiledModule {
            name: metadata_cgu_name,
            kind: ModuleKind::Metadata,
            object: Some(tmp_file),
            dwarf_object: None,
            bytecode: None,
        })
    } else {
        None
    };

    // FIXME handle `-Ctarget-cpu=native`
    let target_cpu = match tcx.sess.opts.cg.target_cpu {
        Some(ref name) => name,
        None => tcx.sess.target.cpu.as_ref(),
    }
    .to_owned();

    Box::new(OngoingCodegen {
        modules,
        allocator_module,
        metadata_module,
        metadata,
        crate_info: CrateInfo::new(tcx, target_cpu),
        concurrency_limiter,
    })
}

// Adapted from https://github.com/rust-lang/rust/blob/303d8aff6092709edd4dbd35b1c88e9aa40bf6d8/src/librustc_codegen_ssa/base.rs#L922-L953
fn determine_cgu_reuse<'tcx>(tcx: TyCtxt<'tcx>, cgu: &CodegenUnit<'tcx>) -> CguReuse {
    if !tcx.dep_graph.is_fully_enabled() {
        return CguReuse::No;
    }

    let work_product_id = &cgu.work_product_id();
    if tcx.dep_graph.previous_work_product(work_product_id).is_none() {
        // We don't have anything cached for this CGU. This can happen
        // if the CGU did not exist in the previous session.
        return CguReuse::No;
    }

    // Try to mark the CGU as green. If it we can do so, it means that nothing
    // affecting the LLVM module has changed and we can re-use a cached version.
    // If we compile with any kind of LTO, this means we can re-use the bitcode
    // of the Pre-LTO stage (possibly also the Post-LTO version but we'll only
    // know that later). If we are not doing LTO, there is only one optimized
    // version of each module, so we re-use that.
    let dep_node = cgu.codegen_dep_node(tcx);
    assert!(
        !tcx.dep_graph.dep_node_exists(&dep_node),
        "CompileCodegenUnit dep-node for CGU `{}` already exists before marking.",
        cgu.name()
    );

    if tcx.try_mark_green(&dep_node) { CguReuse::PostLto } else { CguReuse::No }
}
