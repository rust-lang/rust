//! The AOT driver uses [`cranelift_object`] to write object files suitable for linking into a
//! standalone executable.

use std::fs::File;
use std::io::BufWriter;
use std::path::PathBuf;
use std::sync::Arc;
use std::thread::JoinHandle;

use cranelift_object::{ObjectBuilder, ObjectModule};
use rustc_codegen_ssa::assert_module_sources::CguReuse;
use rustc_codegen_ssa::back::write::produce_final_output_artifacts;
use rustc_codegen_ssa::base::determine_cgu_reuse;
use rustc_codegen_ssa::{CompiledModule, CompiledModules, ModuleKind};
use rustc_data_structures::profiling::SelfProfilerRef;
use rustc_data_structures::stable_hash::{StableHash, StableHashCtxt, StableHasher};
use rustc_data_structures::sync::{IntoDynSyncSend, par_map};
use rustc_hir::attrs::Linkage as RLinkage;
use rustc_middle::dep_graph::{WorkProduct, WorkProductId};
use rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrFlags;
use rustc_middle::mono::{CodegenUnit, MonoItem, MonoItemData, Visibility};
use rustc_session::Session;
use rustc_session::config::{OutputFilenames, OutputType};

use crate::base::CodegenedFunction;
use crate::concurrency_limiter::{ConcurrencyLimiter, ConcurrencyLimiterToken};
use crate::debuginfo::TypeDebugContext;
use crate::global_asm::{GlobalAsmConfig, GlobalAsmContext};
use crate::prelude::*;
use crate::unwind_module::UnwindModule;

struct ModuleCodegenResult {
    module: CompiledModule,
    existing_work_product: Option<(WorkProductId, WorkProduct)>,
}

enum OngoingModuleCodegen {
    Sync(Result<ModuleCodegenResult, String>),
    Async(JoinHandle<Result<ModuleCodegenResult, String>>),
}

impl StableHash for OngoingModuleCodegen {
    fn stable_hash<Hcx: StableHashCtxt>(&self, _: &mut Hcx, _: &mut StableHasher) {
        // do nothing
    }
}

pub(crate) struct OngoingCodegen {
    modules: Vec<OngoingModuleCodegen>,
    allocator_module: Option<CompiledModule>,
    concurrency_limiter: ConcurrencyLimiter,
}

impl OngoingCodegen {
    pub(crate) fn join(
        self,
        sess: &Session,
        outputs: &OutputFilenames,
    ) -> (CompiledModules, FxIndexMap<WorkProductId, WorkProduct>) {
        let mut work_products = FxIndexMap::default();
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
                Err(err) => sess.dcx().fatal(err),
            };
            let ModuleCodegenResult { module, existing_work_product } = module_codegen_result;

            if let Some((work_product_id, work_product)) = existing_work_product {
                work_products.insert(work_product_id, work_product);
            } else {
                let work_product = if sess.opts.unstable_opts.disable_incr_comp_backend_caching {
                    None
                } else if let Some(global_asm_object) = &module.global_asm_object {
                    rustc_incremental::copy_cgu_workproduct_to_incr_comp_cache_dir(
                        sess,
                        &module.name,
                        &[("o", module.object.as_ref().unwrap()), ("asm.o", global_asm_object)],
                        &[],
                    )
                } else {
                    rustc_incremental::copy_cgu_workproduct_to_incr_comp_cache_dir(
                        sess,
                        &module.name,
                        &[("o", module.object.as_ref().unwrap())],
                        &[],
                    )
                };
                if let Some((work_product_id, work_product)) = work_product {
                    work_products.insert(work_product_id, work_product);
                }
            }

            modules.push(module);
        }

        self.concurrency_limiter.finished();

        sess.dcx().abort_if_errors();

        let compiled_modules = CompiledModules { modules, allocator_module: self.allocator_module };

        produce_final_output_artifacts(sess, &compiled_modules, outputs);

        (compiled_modules, work_products)
    }
}

fn make_module(sess: &Session, name: String) -> UnwindModule<ObjectModule> {
    let isa = crate::build_isa(sess, false);

    let mut builder =
        ObjectBuilder::new(isa, name + ".o", cranelift_module::default_libcall_names()).unwrap();

    // Disable function sections by default on MSVC as it causes significant slowdowns with link.exe.
    // Maybe link.exe has exponential behavior when there are many sections with the same name? Also
    // explicitly disable it on MinGW as rustc already disables it by default on MinGW and as such
    // isn't tested. If rustc enables it in the future on MinGW, we can re-enable it too once it has
    // been on MinGW.
    let default_function_sections = sess.target.function_sections && !sess.target.is_like_windows;
    builder.per_function_section(
        sess.opts.unstable_opts.function_sections.unwrap_or(default_function_sections),
    );

    UnwindModule::new(ObjectModule::new(builder), true)
}

fn emit_cgu(
    output_filenames: &OutputFilenames,
    prof: &SelfProfilerRef,
    name: String,
    module: UnwindModule<ObjectModule>,
    debug: Option<DebugContext>,
    global_asm_object_file: Option<PathBuf>,
    producer: &str,
) -> Result<ModuleCodegenResult, String> {
    let mut product = module.finish();

    if let Some(mut debug) = debug {
        debug.emit(&mut product);
    }

    let module = emit_module(
        output_filenames,
        prof,
        product.object,
        ModuleKind::Regular,
        name.clone(),
        global_asm_object_file,
        producer,
    )?;

    Ok(ModuleCodegenResult { module, existing_work_product: None })
}

fn emit_module(
    output_filenames: &OutputFilenames,
    prof: &SelfProfilerRef,
    mut object: cranelift_object::object::write::Object<'_>,
    kind: ModuleKind,
    name: String,
    global_asm_object: Option<PathBuf>,
    producer_str: &str,
) -> Result<CompiledModule, String> {
    if object.format() == cranelift_object::object::BinaryFormat::Elf {
        let comment_section = object.add_section(
            Vec::new(),
            b".comment".to_vec(),
            cranelift_object::object::SectionKind::OtherString,
        );
        let mut producer = vec![0];
        producer.extend(producer_str.as_bytes());
        producer.push(0);
        object.set_section_data(comment_section, producer, 1);
    }

    let tmp_file = output_filenames.temp_path_for_cgu(OutputType::Object, &name);
    let file = match File::create(&tmp_file) {
        Ok(file) => file,
        Err(err) => return Err(format!("error creating object file: {}", err)),
    };

    let mut file = BufWriter::new(file);
    if let Err(err) = object.write_stream(&mut file) {
        return Err(format!("error writing object file: {}", err));
    }
    let file = match file.into_inner() {
        Ok(file) => file,
        Err(err) => return Err(format!("error writing object file: {}", err)),
    };

    if prof.enabled() {
        prof.artifact_size(
            "object_file",
            tmp_file.file_name().unwrap().to_string_lossy(),
            file.metadata().unwrap().len(),
        );
    }

    Ok(CompiledModule {
        name,
        kind,
        object: Some(tmp_file),
        global_asm_object,
        dwarf_object: None,
        bytecode: None,
        assembly: None,
        llvm_ir: None,
        links_from_incr_cache: Vec::new(),
    })
}

fn reuse_workproduct_for_cgu(
    tcx: TyCtxt<'_>,
    cgu: &CodegenUnit<'_>,
) -> Result<ModuleCodegenResult, String> {
    let work_product = cgu.previous_work_product(tcx);
    let obj_out_regular =
        tcx.output_filenames(()).temp_path_for_cgu(OutputType::Object, cgu.name().as_str());
    let source_file_regular = rustc_incremental::in_incr_comp_dir_sess(
        tcx.sess,
        work_product.saved_files.get("o").expect("no saved object file in work product"),
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
        tcx.output_filenames(()).temp_path_ext_for_cgu("asm.o", cgu.name().as_str());
    let source_file_global_asm = if let Some(asm_o) = work_product.saved_files.get("asm.o") {
        let source_file_global_asm = rustc_incremental::in_incr_comp_dir_sess(tcx.sess, asm_o);
        if let Err(err) = rustc_fs_util::link_or_copy(&source_file_global_asm, &obj_out_global_asm)
        {
            return Err(format!(
                "unable to copy {} to {}: {}",
                source_file_global_asm.display(),
                obj_out_global_asm.display(),
                err
            ));
        }
        Some(source_file_global_asm)
    } else {
        None
    };

    Ok(ModuleCodegenResult {
        module: CompiledModule {
            name: cgu.name().to_string(),
            kind: ModuleKind::Regular,
            object: Some(obj_out_regular),
            global_asm_object: source_file_global_asm.as_ref().map(|_| obj_out_global_asm),
            dwarf_object: None,
            bytecode: None,
            assembly: None,
            llvm_ir: None,
            links_from_incr_cache: if let Some(source_file_global_asm) = source_file_global_asm {
                vec![source_file_regular, source_file_global_asm]
            } else {
                vec![source_file_regular]
            },
        },
        existing_work_product: Some((cgu.work_product_id(), work_product)),
    })
}

fn codegen_cgu_content(
    tcx: TyCtxt<'_>,
    module: &mut dyn Module,
    cgu_name: rustc_span::Symbol,
) -> (Option<DebugContext>, Vec<CodegenedFunction>, String) {
    let _timer = tcx.prof.generic_activity_with_arg("codegen cgu", cgu_name.as_str());

    let cgu = tcx.codegen_unit(cgu_name);
    let mono_items = cgu.items_in_deterministic_order(tcx);

    let mut debug_context = DebugContext::new(tcx, module.isa(), false, cgu_name.as_str());
    let mut global_asm = String::new();
    let mut type_dbg = TypeDebugContext::default();
    super::predefine_mono_items(tcx, module, &mono_items);
    let mut codegened_functions = vec![];
    for (mono_item, item_data) in mono_items {
        match mono_item {
            MonoItem::Fn(instance) => {
                let flags = tcx.codegen_instance_attrs(instance.def).flags;
                if flags.contains(CodegenFnAttrFlags::NAKED) {
                    rustc_codegen_ssa::mir::naked_asm::codegen_naked_asm(
                        &mut GlobalAsmContext { tcx, global_asm: &mut global_asm },
                        instance,
                        MonoItemData {
                            linkage: RLinkage::External,
                            visibility: if item_data.linkage == RLinkage::Internal {
                                Visibility::Hidden
                            } else {
                                item_data.visibility
                            },
                            ..item_data
                        },
                    );
                    continue;
                }
                let codegened_function = crate::base::codegen_fn(
                    tcx,
                    cgu_name,
                    debug_context.as_mut(),
                    &mut type_dbg,
                    Function::new(),
                    module,
                    instance,
                );
                codegened_functions.push(codegened_function);
            }
            MonoItem::Static(def_id) => {
                let data_id = crate::constant::codegen_static(tcx, module, def_id);
                if let Some(debug_context) = debug_context.as_mut() {
                    debug_context.define_static(tcx, &mut type_dbg, def_id, data_id);
                }
            }
            MonoItem::GlobalAsm(item_id) => {
                rustc_codegen_ssa::base::codegen_global_asm(
                    &mut GlobalAsmContext { tcx, global_asm: &mut global_asm },
                    item_id,
                );
            }
        }
    }
    crate::main_shim::maybe_create_entry_wrapper(tcx, module, false, cgu.is_primary());

    (debug_context, codegened_functions, global_asm)
}

fn module_codegen(
    tcx: TyCtxt<'_>,
    global_asm_config: Arc<GlobalAsmConfig>,
    cgu_name: rustc_span::Symbol,
    token: ConcurrencyLimiterToken,
) -> OngoingModuleCodegen {
    let mut module = make_module(tcx.sess, cgu_name.as_str().to_string());

    let (mut debug_context, codegened_functions, mut global_asm) =
        codegen_cgu_content(tcx, &mut module, cgu_name);

    let cgu_name = cgu_name.as_str().to_owned();

    let producer = crate::debuginfo::producer(tcx.sess);

    let profiler = tcx.prof.clone();
    let output_filenames = tcx.output_filenames(()).clone();
    let should_write_ir = crate::pretty_clif::should_write_ir(tcx.sess);

    OngoingModuleCodegen::Async(std::thread::spawn(move || {
        profiler.clone().generic_activity_with_arg("compile functions", &*cgu_name).run(|| {
            cranelift_codegen::timing::set_thread_profiler(Box::new(super::MeasuremeProfiler(
                profiler.clone(),
            )));

            let mut cached_context = Context::new();
            for codegened_func in codegened_functions {
                crate::base::compile_fn(
                    &profiler,
                    &output_filenames,
                    should_write_ir,
                    &mut cached_context,
                    &mut module,
                    debug_context.as_mut(),
                    &mut global_asm,
                    codegened_func,
                );
            }
        });

        let global_asm_object_file =
            profiler.generic_activity_with_arg("compile assembly", &*cgu_name).run(|| {
                if global_asm.is_empty() {
                    return Ok::<_, String>(None);
                }

                let global_asm_object_file =
                    output_filenames.temp_path_ext_for_cgu("asm.o", &*cgu_name);
                crate::global_asm::compile_global_asm(
                    &global_asm_config,
                    global_asm,
                    &global_asm_object_file,
                )?;

                Ok(Some(global_asm_object_file))
            })?;

        let codegen_result =
            profiler.generic_activity_with_arg("write object file", &*cgu_name).run(|| {
                emit_cgu(
                    &output_filenames,
                    &profiler,
                    cgu_name,
                    module,
                    debug_context,
                    global_asm_object_file,
                    &producer,
                )
            });
        std::mem::drop(token);
        codegen_result
    }))
}

fn emit_allocator_module(tcx: TyCtxt<'_>) -> Option<CompiledModule> {
    let mut allocator_module = make_module(tcx.sess, "allocator_shim".to_string());
    let created_alloc_shim = crate::allocator::codegen(tcx, &mut allocator_module);

    if created_alloc_shim {
        let product = allocator_module.finish();

        match emit_module(
            tcx.output_filenames(()),
            &tcx.sess.prof,
            product.object,
            ModuleKind::Allocator,
            "allocator_shim".to_owned(),
            None,
            &crate::debuginfo::producer(tcx.sess),
        ) {
            Ok(allocator_module) => Some(allocator_module),
            Err(err) => tcx.dcx().fatal(err),
        }
    } else {
        None
    }
}

pub(crate) fn run_aot(tcx: TyCtxt<'_>) -> Box<OngoingCodegen> {
    let cgus = tcx.collect_and_partition_mono_items(()).codegen_units;

    if tcx.dep_graph.is_fully_enabled() {
        for cgu in cgus {
            tcx.ensure_ok().codegen_unit(cgu.name());
        }
    }

    // Calculate the CGU reuse
    let cgu_reuse = tcx.sess.time("find_cgu_reuse", || {
        cgus.iter().map(|cgu| determine_cgu_reuse(tcx, cgu)).collect::<Vec<_>>()
    });

    rustc_codegen_ssa::assert_module_sources::assert_module_sources(tcx, &|cgu_reuse_tracker| {
        for (i, cgu) in cgus.iter().enumerate() {
            let cgu_reuse = cgu_reuse[i];
            cgu_reuse_tracker.set_actual_reuse(cgu.name().as_str(), cgu_reuse);
        }
    });

    let global_asm_config = Arc::new(crate::global_asm::GlobalAsmConfig::new(tcx.sess));

    let (todo_cgus, done_cgus) =
        cgus.iter().enumerate().partition::<Vec<_>, _>(|&(i, _)| match cgu_reuse[i] {
            _ if tcx.sess.opts.unstable_opts.disable_incr_comp_backend_caching => true,
            CguReuse::No => true,
            CguReuse::PreLto | CguReuse::PostLto => false,
        });

    let concurrency_limiter = IntoDynSyncSend(ConcurrencyLimiter::new(todo_cgus.len()));

    let modules: Vec<_> =
        tcx.sess.time("codegen mono items", || {
            let modules: Vec<_> = par_map(todo_cgus, |(_, cgu)| {
                let dep_node = cgu.codegen_dep_node(tcx);
                let (module, _) = tcx.dep_graph.with_task(
                    dep_node,
                    tcx,
                    || {
                        module_codegen(
                            tcx,
                            global_asm_config.clone(),
                            cgu.name(),
                            concurrency_limiter.acquire(tcx.dcx()),
                        )
                    },
                    Some(rustc_middle::dep_graph::hash_result),
                );
                IntoDynSyncSend(module)
            });
            modules
                .into_iter()
                .map(|module| module.0)
                .chain(done_cgus.into_iter().map(|(_, cgu)| {
                    OngoingModuleCodegen::Sync(reuse_workproduct_for_cgu(tcx, cgu))
                }))
                .collect()
        });

    let allocator_module = emit_allocator_module(tcx);

    Box::new(OngoingCodegen {
        modules,
        allocator_module,
        concurrency_limiter: concurrency_limiter.0,
    })
}
