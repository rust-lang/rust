//! The AOT driver uses [`cranelift_object`] to write object files suitable for linking into a
//! standalone executable.

use std::env;
use std::fs::File;
use std::io::BufWriter;
use std::path::PathBuf;
use std::sync::Arc;
use std::thread::JoinHandle;

use cranelift_object::{ObjectBuilder, ObjectModule};
use rustc_codegen_ssa::assert_module_sources::CguReuse;
use rustc_codegen_ssa::back::write::{CompiledModules, produce_final_output_artifacts};
use rustc_codegen_ssa::base::determine_cgu_reuse;
use rustc_codegen_ssa::{CodegenResults, CompiledModule, CrateInfo, ModuleKind};
use rustc_data_structures::profiling::SelfProfilerRef;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_data_structures::sync::{IntoDynSyncSend, par_map};
use rustc_hir::attrs::Linkage as RLinkage;
use rustc_middle::dep_graph::{WorkProduct, WorkProductId};
use rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrFlags;
use rustc_middle::mir::mono::{CodegenUnit, MonoItem, MonoItemData, Visibility};
use rustc_session::Session;
use rustc_session::config::{OutputFilenames, OutputType};

use crate::base::CodegenedFunction;
use crate::concurrency_limiter::{ConcurrencyLimiter, ConcurrencyLimiterToken};
use crate::debuginfo::TypeDebugContext;
use crate::global_asm::{GlobalAsmConfig, GlobalAsmContext};
use crate::prelude::*;
use crate::unwind_module::UnwindModule;

fn disable_incr_cache() -> bool {
    env::var("CG_CLIF_DISABLE_INCR_CACHE").as_deref() == Ok("1")
}

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
    crate_info: CrateInfo,
    concurrency_limiter: ConcurrencyLimiter,
}

impl OngoingCodegen {
    pub(crate) fn join(
        self,
        sess: &Session,
        outputs: &OutputFilenames,
    ) -> (CodegenResults, FxIndexMap<WorkProductId, WorkProduct>) {
        let mut work_products = FxIndexMap::default();
        let mut modules = vec![];
        let disable_incr_cache = disable_incr_cache();

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
            let ModuleCodegenResult { module_regular, module_global_asm, existing_work_product } =
                module_codegen_result;

            if let Some((work_product_id, work_product)) = existing_work_product {
                work_products.insert(work_product_id, work_product);
            } else {
                let work_product = if disable_incr_cache {
                    None
                } else if let Some(module_global_asm) = &module_global_asm {
                    rustc_incremental::copy_cgu_workproduct_to_incr_comp_cache_dir(
                        sess,
                        &module_regular.name,
                        &[
                            ("o", module_regular.object.as_ref().unwrap()),
                            ("asm.o", module_global_asm.object.as_ref().unwrap()),
                        ],
                        &[],
                    )
                } else {
                    rustc_incremental::copy_cgu_workproduct_to_incr_comp_cache_dir(
                        sess,
                        &module_regular.name,
                        &[("o", module_regular.object.as_ref().unwrap())],
                        &[],
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

        sess.dcx().abort_if_errors();

        let compiled_modules = CompiledModules { modules, allocator_module: self.allocator_module };

        produce_final_output_artifacts(sess, &compiled_modules, outputs);

        (
            CodegenResults {
                crate_info: self.crate_info,

                modules: compiled_modules.modules,
                allocator_module: compiled_modules.allocator_module,
            },
            work_products,
        )
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
    invocation_temp: Option<&str>,
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

    let module_regular = emit_module(
        output_filenames,
        invocation_temp,
        prof,
        product.object,
        ModuleKind::Regular,
        name.clone(),
        producer,
    )?;

    Ok(ModuleCodegenResult {
        module_regular,
        module_global_asm: global_asm_object_file.map(|global_asm_object_file| CompiledModule {
            name: format!("{name}.asm"),
            kind: ModuleKind::Regular,
            object: Some(global_asm_object_file),
            dwarf_object: None,
            bytecode: None,
            assembly: None,
            llvm_ir: None,
            links_from_incr_cache: Vec::new(),
        }),
        existing_work_product: None,
    })
}

fn emit_module(
    output_filenames: &OutputFilenames,
    invocation_temp: Option<&str>,
    prof: &SelfProfilerRef,
    mut object: cranelift_object::object::write::Object<'_>,
    kind: ModuleKind,
    name: String,
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

    let tmp_file = output_filenames.temp_path_for_cgu(OutputType::Object, &name, invocation_temp);
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
    let obj_out_regular = tcx.output_filenames(()).temp_path_for_cgu(
        OutputType::Object,
        cgu.name().as_str(),
        tcx.sess.invocation_temp.as_deref(),
    );
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
        crate::global_asm::add_file_stem_postfix(obj_out_regular.clone(), ".asm");
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
        module_regular: CompiledModule {
            name: cgu.name().to_string(),
            kind: ModuleKind::Regular,
            object: Some(obj_out_regular),
            dwarf_object: None,
            bytecode: None,
            assembly: None,
            llvm_ir: None,
            links_from_incr_cache: vec![source_file_regular],
        },
        module_global_asm: source_file_global_asm.map(|source_file| CompiledModule {
            name: cgu.name().to_string(),
            kind: ModuleKind::Regular,
            object: Some(obj_out_global_asm),
            dwarf_object: None,
            bytecode: None,
            assembly: None,
            llvm_ir: None,
            links_from_incr_cache: vec![source_file],
        }),
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
    (global_asm_config, cgu_name, token): (
        Arc<GlobalAsmConfig>,
        rustc_span::Symbol,
        ConcurrencyLimiterToken,
    ),
) -> OngoingModuleCodegen {
    let mut module = make_module(tcx.sess, cgu_name.as_str().to_string());

    let (mut debug_context, codegened_functions, mut global_asm) =
        codegen_cgu_content(tcx, &mut module, cgu_name);

    let cgu_name = cgu_name.as_str().to_owned();

    let producer = crate::debuginfo::producer(tcx.sess);

    let profiler = tcx.prof.clone();
    let invocation_temp = tcx.sess.invocation_temp.clone();
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
                crate::global_asm::compile_global_asm(
                    &global_asm_config,
                    &cgu_name,
                    global_asm,
                    invocation_temp.as_deref(),
                )
            })?;

        let codegen_result =
            profiler.generic_activity_with_arg("write object file", &*cgu_name).run(|| {
                emit_cgu(
                    &global_asm_config.output_filenames,
                    invocation_temp.as_deref(),
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
            tcx.sess.invocation_temp.as_deref(),
            &tcx.sess.prof,
            product.object,
            ModuleKind::Allocator,
            "allocator_shim".to_owned(),
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
    // FIXME handle `-Ctarget-cpu=native`
    let target_cpu = match tcx.sess.opts.cg.target_cpu {
        Some(ref name) => name,
        None => tcx.sess.target.cpu.as_ref(),
    }
    .to_owned();

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

    let global_asm_config = Arc::new(crate::global_asm::GlobalAsmConfig::new(tcx));

    let disable_incr_cache = disable_incr_cache();
    let (todo_cgus, done_cgus) =
        cgus.iter().enumerate().partition::<Vec<_>, _>(|&(i, _)| match cgu_reuse[i] {
            _ if disable_incr_cache => true,
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
                    (global_asm_config.clone(), cgu.name(), concurrency_limiter.acquire(tcx.dcx())),
                    module_codegen,
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
        crate_info: CrateInfo::new(tcx, target_cpu),
        concurrency_limiter: concurrency_limiter.0,
    })
}
