use rustc::dep_graph::{WorkProduct, WorkProductFileKind, WorkProductId};
use rustc::middle::cstore::EncodedMetadata;
use rustc::mir::mono::CodegenUnit;
use rustc::session::config::{DebugInfo, OutputType};
use rustc_session::cgu_reuse_tracker::CguReuse;
use rustc_codegen_ssa::back::linker::LinkerInfo;
use rustc_codegen_ssa::CrateInfo;

use crate::prelude::*;

use crate::backend::{Emit, WriteDebugInfo};

pub(super) fn run_aot(
    tcx: TyCtxt<'_>,
    metadata: EncodedMetadata,
    need_metadata_module: bool,
) -> Box<(CodegenResults, FxHashMap<WorkProductId, WorkProduct>)> {
    let mut work_products = FxHashMap::default();

    fn new_module(tcx: TyCtxt<'_>, name: String) -> Module<crate::backend::Backend> {
        let module = crate::backend::make_module(tcx.sess, name);
        assert_eq!(pointer_ty(tcx), module.target_config().pointer_type());
        module
    };

    struct ModuleCodegenResult(CompiledModule, Option<(WorkProductId, WorkProduct)>);

    use rustc_data_structures::stable_hasher::{HashStable, StableHasher};

    impl<HCX> HashStable<HCX> for ModuleCodegenResult {
        fn hash_stable(&self, _: &mut HCX, _: &mut StableHasher) {
            // do nothing
        }
    }

    fn emit_module<B: Backend>(
        tcx: TyCtxt<'_>,
        name: String,
        kind: ModuleKind,
        mut module: Module<B>,
        debug: Option<DebugContext>,
    ) -> ModuleCodegenResult
        where B::Product: Emit + WriteDebugInfo,
    {
            module.finalize_definitions();
            let mut product = module.finish();

            if let Some(mut debug) = debug {
                debug.emit(&mut product);
            }

            let tmp_file = tcx
                .output_filenames(LOCAL_CRATE)
                .temp_path(OutputType::Object, Some(&name));
            let obj = product.emit();
            std::fs::write(&tmp_file, obj).unwrap();

            let work_product = if std::env::var("CG_CLIF_INCR_CACHE_DISABLED").is_ok() {
                None
            } else {
                rustc_incremental::copy_cgu_workproducts_to_incr_comp_cache_dir(
                    tcx.sess,
                    &name,
                    &[(WorkProductFileKind::Object, tmp_file.clone())],
                )
            };

            ModuleCodegenResult(
                CompiledModule {
                    name,
                    kind,
                    object: Some(tmp_file),
                    bytecode: None,
                    bytecode_compressed: None,
                },
                work_product,
            )
        };

    let (_, cgus) = tcx.collect_and_partition_mono_items(LOCAL_CRATE);

    if tcx.dep_graph.is_fully_enabled() {
        for cgu in &*cgus {
            tcx.codegen_unit(cgu.name());
        }
    }

    let modules = super::time(tcx, "codegen mono items", || {
        cgus.iter().map(|cgu| {
            let cgu_reuse = determine_cgu_reuse(tcx, cgu);
            tcx.sess.cgu_reuse_tracker.set_actual_reuse(&cgu.name().as_str(), cgu_reuse);

            match cgu_reuse {
                _ if std::env::var("CG_CLIF_INCR_CACHE_DISABLED").is_ok() => {}
                CguReuse::No => {}
                CguReuse::PreLto => {
                    let incr_comp_session_dir = tcx.sess.incr_comp_session_dir();
                    let mut object = None;
                    let work_product = cgu.work_product(tcx);
                    for (kind, saved_file) in &work_product.saved_files {
                        let obj_out = match kind {
                            WorkProductFileKind::Object => {
                                let path = tcx.output_filenames(LOCAL_CRATE).temp_path(OutputType::Object, Some(&cgu.name().as_str()));
                                object = Some(path.clone());
                                path
                            }
                            WorkProductFileKind::Bytecode | WorkProductFileKind::BytecodeCompressed => {
                                panic!("cg_clif doesn't use bytecode");
                            }
                        };
                        let source_file = rustc_incremental::in_incr_comp_dir(&incr_comp_session_dir, &saved_file);
                        if let Err(err) = rustc_fs_util::link_or_copy(&source_file, &obj_out) {
                            tcx.sess.err(&format!(
                                "unable to copy {} to {}: {}",
                                source_file.display(),
                                obj_out.display(),
                                err
                            ));
                        }
                    }

                    work_products.insert(cgu.work_product_id(), work_product);

                    return CompiledModule {
                        name: cgu.name().to_string(),
                        kind: ModuleKind::Regular,
                        object,
                        bytecode: None,
                        bytecode_compressed: None,
                    };
                }
                CguReuse::PostLto => unreachable!(),
            }

            let dep_node = cgu.codegen_dep_node(tcx);
            let (ModuleCodegenResult(module, work_product), _) =
                tcx.dep_graph.with_task(dep_node, tcx, cgu.name(), module_codegen, rustc::dep_graph::hash_result);

            fn module_codegen(tcx: TyCtxt<'_>, cgu_name: rustc_span::Symbol) -> ModuleCodegenResult {
                let cgu = tcx.codegen_unit(cgu_name);
                let mono_items = cgu.items_in_deterministic_order(tcx);

                let mut module = new_module(tcx, cgu_name.as_str().to_string());

                let mut debug = if tcx.sess.opts.debuginfo != DebugInfo::None {
                    let debug = DebugContext::new(
                        tcx,
                        module.target_config().pointer_type().bytes() as u8,
                    );
                    Some(debug)
                } else {
                    None
                };

                super::codegen_mono_items(tcx, &mut module, debug.as_mut(), mono_items);
                crate::main_shim::maybe_create_entry_wrapper(tcx, &mut module);

                emit_module(
                    tcx,
                    cgu.name().as_str().to_string(),
                    ModuleKind::Regular,
                    module,
                    debug,
                )
            }

            if let Some((id, product)) = work_product {
                work_products.insert(id, product);
            }

            module
        }).collect::<Vec<_>>()
    });

    tcx.sess.abort_if_errors();

    let mut allocator_module = new_module(tcx, "allocator_shim".to_string());
    let created_alloc_shim = crate::allocator::codegen(tcx, &mut allocator_module);

    let allocator_module = if created_alloc_shim {
        let ModuleCodegenResult(module, work_product) = emit_module(
            tcx,
            "allocator_shim".to_string(),
            ModuleKind::Allocator,
            allocator_module,
            None,
        );
        if let Some((id, product)) = work_product {
            work_products.insert(id, product);
        }
        Some(module)
    } else {
        None
    };

    rustc_incremental::assert_dep_graph(tcx);
    rustc_incremental::save_dep_graph(tcx);

    let metadata_module = if need_metadata_module {
        let _timer = tcx.prof.generic_activity("codegen crate metadata");
        let (metadata_cgu_name, tmp_file) = tcx.sess.time("write compressed metadata", || {
            use rustc::mir::mono::CodegenUnitNameBuilder;

            let cgu_name_builder = &mut CodegenUnitNameBuilder::new(tcx);
            let metadata_cgu_name = cgu_name_builder
                .build_cgu_name(LOCAL_CRATE, &["crate"], Some("metadata"))
                .as_str()
                .to_string();

            let tmp_file = tcx
                .output_filenames(LOCAL_CRATE)
                .temp_path(OutputType::Metadata, Some(&metadata_cgu_name));

            let obj = crate::backend::with_object(tcx.sess, &metadata_cgu_name, |object| {
                crate::metadata::write_metadata(tcx, object);
            });

            std::fs::write(&tmp_file, obj).unwrap();

            (metadata_cgu_name, tmp_file)
        });

        Some(CompiledModule {
            name: metadata_cgu_name,
            kind: ModuleKind::Metadata,
            object: Some(tmp_file),
            bytecode: None,
            bytecode_compressed: None,
        })
    } else {
        None
    };

    Box::new((CodegenResults {
        crate_name: tcx.crate_name(LOCAL_CRATE),
        modules,
        allocator_module,
        metadata_module,
        crate_hash: tcx.crate_hash(LOCAL_CRATE),
        metadata,
        windows_subsystem: None, // Windows is not yet supported
        linker_info: LinkerInfo::new(tcx),
        crate_info: CrateInfo::new(tcx),
    }, work_products))
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

    if tcx.dep_graph.try_mark_green(tcx, &dep_node).is_some() {
        CguReuse::PreLto
    } else {
        CguReuse::No
    }
}
