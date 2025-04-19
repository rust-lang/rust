//! The AOT driver uses [`cranelift_object`] to write object files suitable for linking into a
//! standalone executable.

use std::env;
use std::fs::{self, File};
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread::JoinHandle;

use cranelift_object::{ObjectBuilder, ObjectModule};
use rustc_codegen_ssa::assert_module_sources::CguReuse;
use rustc_codegen_ssa::back::link::ensure_removed;
use rustc_codegen_ssa::back::metadata::create_compressed_metadata_file;
use rustc_codegen_ssa::base::determine_cgu_reuse;
use rustc_codegen_ssa::{
    CodegenResults, CompiledModule, CrateInfo, ModuleKind, errors as ssa_errors,
};
use rustc_data_structures::profiling::SelfProfilerRef;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_data_structures::sync::{IntoDynSyncSend, par_map};
use rustc_metadata::EncodedMetadata;
use rustc_metadata::fs::copy_to_stdout;
use rustc_middle::dep_graph::{WorkProduct, WorkProductId};
use rustc_middle::mir::mono::{CodegenUnit, MonoItem};
use rustc_session::Session;
use rustc_session::config::{DebugInfo, OutFileName, OutputFilenames, OutputType};

use crate::CodegenCx;
use crate::base::CodegenedFunction;
use crate::concurrency_limiter::{ConcurrencyLimiter, ConcurrencyLimiterToken};
use crate::debuginfo::TypeDebugContext;
use crate::global_asm::GlobalAsmConfig;
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
    metadata_module: Option<CompiledModule>,
    metadata: EncodedMetadata,
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
                            ("o", &module_regular.object.as_ref().unwrap()),
                            ("asm.o", &module_global_asm.object.as_ref().unwrap()),
                        ],
                        &[],
                    )
                } else {
                    rustc_incremental::copy_cgu_workproduct_to_incr_comp_cache_dir(
                        sess,
                        &module_regular.name,
                        &[("o", &module_regular.object.as_ref().unwrap())],
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

        let codegen_results = CodegenResults {
            modules,
            allocator_module: self.allocator_module,
            metadata_module: self.metadata_module,
            metadata: self.metadata,
            crate_info: self.crate_info,
        };

        produce_final_output_artifacts(sess, &codegen_results, outputs);

        (codegen_results, work_products)
    }
}

// Adapted from https://github.com/rust-lang/rust/blob/73476d49904751f8d90ce904e16dfbc278083d2c/compiler/rustc_codegen_ssa/src/back/write.rs#L547C1-L706C2
fn produce_final_output_artifacts(
    sess: &Session,
    codegen_results: &CodegenResults,
    crate_output: &OutputFilenames,
) {
    let user_wants_bitcode = false;
    let mut user_wants_objects = false;

    // Produce final compile outputs.
    let copy_gracefully = |from: &Path, to: &OutFileName| match to {
        OutFileName::Stdout => {
            if let Err(e) = copy_to_stdout(from) {
                sess.dcx().emit_err(ssa_errors::CopyPath::new(from, to.as_path(), e));
            }
        }
        OutFileName::Real(path) => {
            if let Err(e) = fs::copy(from, path) {
                sess.dcx().emit_err(ssa_errors::CopyPath::new(from, path, e));
            }
        }
    };

    let copy_if_one_unit = |output_type: OutputType, keep_numbered: bool| {
        if codegen_results.modules.len() == 1 {
            // 1) Only one codegen unit. In this case it's no difficulty
            //    to copy `foo.0.x` to `foo.x`.
            let path = crate_output.temp_path_for_cgu(
                output_type,
                &codegen_results.modules[0].name,
                sess.invocation_temp.as_deref(),
            );
            let output = crate_output.path(output_type);
            if !output_type.is_text_output() && output.is_tty() {
                sess.dcx()
                    .emit_err(ssa_errors::BinaryOutputToTty { shorthand: output_type.shorthand() });
            } else {
                copy_gracefully(&path, &output);
            }
            if !sess.opts.cg.save_temps && !keep_numbered {
                // The user just wants `foo.x`, not `foo.#module-name#.x`.
                ensure_removed(sess.dcx(), &path);
            }
        } else {
            if crate_output.outputs.contains_explicit_name(&output_type) {
                // 2) Multiple codegen units, with `--emit foo=some_name`. We have
                //    no good solution for this case, so warn the user.
                sess.dcx()
                    .emit_warn(ssa_errors::IgnoringEmitPath { extension: output_type.extension() });
            } else if crate_output.single_output_file.is_some() {
                // 3) Multiple codegen units, with `-o some_name`. We have
                //    no good solution for this case, so warn the user.
                sess.dcx()
                    .emit_warn(ssa_errors::IgnoringOutput { extension: output_type.extension() });
            } else {
                // 4) Multiple codegen units, but no explicit name. We
                //    just leave the `foo.0.x` files in place.
                // (We don't have to do any work in this case.)
            }
        }
    };

    // Flag to indicate whether the user explicitly requested bitcode.
    // Otherwise, we produced it only as a temporary output, and will need
    // to get rid of it.
    for output_type in crate_output.outputs.keys() {
        match *output_type {
            OutputType::Bitcode | OutputType::ThinLinkBitcode => {
                // Cranelift doesn't have bitcode
                // user_wants_bitcode = true;
                // // Copy to .bc, but always keep the .0.bc. There is a later
                // // check to figure out if we should delete .0.bc files, or keep
                // // them for making an rlib.
                // copy_if_one_unit(OutputType::Bitcode, true);
            }
            OutputType::LlvmAssembly => {
                // Cranelift IR text already emitted during codegen
                // copy_if_one_unit(OutputType::LlvmAssembly, false);
            }
            OutputType::Assembly => {
                // Currently no support for emitting raw assembly files
                // copy_if_one_unit(OutputType::Assembly, false);
            }
            OutputType::Object => {
                user_wants_objects = true;
                copy_if_one_unit(OutputType::Object, true);
            }
            OutputType::Mir | OutputType::Metadata | OutputType::Exe | OutputType::DepInfo => {}
        }
    }

    // Clean up unwanted temporary files.

    // We create the following files by default:
    //  - #crate#.#module-name#.bc
    //  - #crate#.#module-name#.o
    //  - #crate#.crate.metadata.bc
    //  - #crate#.crate.metadata.o
    //  - #crate#.o (linked from crate.##.o)
    //  - #crate#.bc (copied from crate.##.bc)
    // We may create additional files if requested by the user (through
    // `-C save-temps` or `--emit=` flags).

    if !sess.opts.cg.save_temps {
        // Remove the temporary .#module-name#.o objects. If the user didn't
        // explicitly request bitcode (with --emit=bc), and the bitcode is not
        // needed for building an rlib, then we must remove .#module-name#.bc as
        // well.

        // Specific rules for keeping .#module-name#.bc:
        //  - If the user requested bitcode (`user_wants_bitcode`), and
        //    codegen_units > 1, then keep it.
        //  - If the user requested bitcode but codegen_units == 1, then we
        //    can toss .#module-name#.bc because we copied it to .bc earlier.
        //  - If we're not building an rlib and the user didn't request
        //    bitcode, then delete .#module-name#.bc.
        // If you change how this works, also update back::link::link_rlib,
        // where .#module-name#.bc files are (maybe) deleted after making an
        // rlib.
        let needs_crate_object = crate_output.outputs.contains_key(&OutputType::Exe);

        let keep_numbered_bitcode = user_wants_bitcode && sess.codegen_units().as_usize() > 1;

        let keep_numbered_objects =
            needs_crate_object || (user_wants_objects && sess.codegen_units().as_usize() > 1);

        for module in codegen_results.modules.iter() {
            if let Some(ref path) = module.object {
                if !keep_numbered_objects {
                    ensure_removed(sess.dcx(), path);
                }
            }

            if let Some(ref path) = module.dwarf_object {
                if !keep_numbered_objects {
                    ensure_removed(sess.dcx(), path);
                }
            }

            if let Some(ref path) = module.bytecode {
                if !keep_numbered_bitcode {
                    ensure_removed(sess.dcx(), path);
                }
            }
        }

        if !user_wants_bitcode {
            if let Some(ref allocator_module) = codegen_results.allocator_module {
                if let Some(ref path) = allocator_module.bytecode {
                    ensure_removed(sess.dcx(), path);
                }
            }
        }
    }

    if sess.opts.json_artifact_notifications {
        if codegen_results.modules.len() == 1 {
            codegen_results.modules[0].for_each_output(|_path, ty| {
                if sess.opts.output_types.contains_key(&ty) {
                    let descr = ty.shorthand();
                    // for single cgu file is renamed to drop cgu specific suffix
                    // so we regenerate it the same way
                    let path = crate_output.path(ty);
                    sess.dcx().emit_artifact_notification(path.as_path(), descr);
                }
            });
        } else {
            for module in &codegen_results.modules {
                module.for_each_output(|path, ty| {
                    if sess.opts.output_types.contains_key(&ty) {
                        let descr = ty.shorthand();
                        sess.dcx().emit_artifact_notification(&path, descr);
                    }
                });
            }
        }
    }

    // We leave the following files around by default:
    //  - #crate#.o
    //  - #crate#.crate.metadata.o
    //  - #crate#.bc
    // These are used in linking steps and will be cleaned up afterward.
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
    let source_file_global_asm = if let Some(asm_o) = work_product.saved_files.get("asm.o") {
        let source_file_global_asm = rustc_incremental::in_incr_comp_dir_sess(&tcx.sess, asm_o);
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
) -> (CodegenCx, Vec<CodegenedFunction>) {
    let _timer = tcx.prof.generic_activity_with_arg("codegen cgu", cgu_name.as_str());

    let cgu = tcx.codegen_unit(cgu_name);
    let mono_items = cgu.items_in_deterministic_order(tcx);

    let mut cx = crate::CodegenCx::new(
        tcx,
        module.isa(),
        tcx.sess.opts.debuginfo != DebugInfo::None,
        cgu_name,
    );
    let mut type_dbg = TypeDebugContext::default();
    super::predefine_mono_items(tcx, module, &mono_items);
    let mut codegened_functions = vec![];
    for (mono_item, _) in mono_items {
        match mono_item {
            MonoItem::Fn(inst) => {
                if let Some(codegened_function) = crate::base::codegen_fn(
                    tcx,
                    &mut cx,
                    &mut type_dbg,
                    Function::new(),
                    module,
                    inst,
                ) {
                    codegened_functions.push(codegened_function);
                }
            }
            MonoItem::Static(def_id) => {
                let data_id = crate::constant::codegen_static(tcx, module, def_id);
                if let Some(debug_context) = &mut cx.debug_context {
                    debug_context.define_static(tcx, &mut type_dbg, def_id, data_id);
                }
            }
            MonoItem::GlobalAsm(item_id) => {
                crate::global_asm::codegen_global_asm_item(tcx, &mut cx.global_asm, item_id);
            }
        }
    }
    crate::main_shim::maybe_create_entry_wrapper(tcx, module, false, cgu.is_primary());

    (cx, codegened_functions)
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

    let (mut cx, codegened_functions) = codegen_cgu_content(tcx, &mut module, cgu_name);

    let cgu_name = cgu_name.as_str().to_owned();

    let producer = crate::debuginfo::producer(tcx.sess);

    let profiler = tcx.prof.clone();

    OngoingModuleCodegen::Async(std::thread::spawn(move || {
        profiler.clone().generic_activity_with_arg("compile functions", &*cgu_name).run(|| {
            cranelift_codegen::timing::set_thread_profiler(Box::new(super::MeasuremeProfiler(
                profiler.clone(),
            )));

            let mut cached_context = Context::new();
            for codegened_func in codegened_functions {
                crate::base::compile_fn(
                    &mut cx,
                    &profiler,
                    &mut cached_context,
                    &mut module,
                    codegened_func,
                );
            }
        });

        let global_asm_object_file =
            profiler.generic_activity_with_arg("compile assembly", &*cgu_name).run(|| {
                crate::global_asm::compile_global_asm(
                    &global_asm_config,
                    &cgu_name,
                    &cx.global_asm,
                    cx.invocation_temp.as_deref(),
                )
            })?;

        let codegen_result =
            profiler.generic_activity_with_arg("write object file", &*cgu_name).run(|| {
                emit_cgu(
                    &global_asm_config.output_filenames,
                    cx.invocation_temp.as_deref(),
                    &profiler,
                    cgu_name,
                    module,
                    cx.debug_context,
                    global_asm_object_file,
                    &producer,
                )
            });
        std::mem::drop(token);
        codegen_result
    }))
}

fn emit_metadata_module(tcx: TyCtxt<'_>, metadata: &EncodedMetadata) -> CompiledModule {
    use rustc_middle::mir::mono::CodegenUnitNameBuilder;

    let _timer = tcx.sess.timer("write compressed metadata");

    let cgu_name_builder = &mut CodegenUnitNameBuilder::new(tcx);
    let metadata_cgu_name = cgu_name_builder
        .build_cgu_name(LOCAL_CRATE, ["crate"], Some("metadata"))
        .as_str()
        .to_string();

    let tmp_file = tcx.output_filenames(()).temp_path_for_cgu(
        OutputType::Metadata,
        &metadata_cgu_name,
        tcx.sess.invocation_temp.as_deref(),
    );

    let symbol_name = rustc_middle::middle::exported_symbols::metadata_symbol_name(tcx);
    let obj = create_compressed_metadata_file(tcx.sess, metadata, &symbol_name);

    if let Err(err) = std::fs::write(&tmp_file, obj) {
        tcx.dcx().fatal(format!("error writing metadata object file: {}", err));
    }

    CompiledModule {
        name: metadata_cgu_name,
        kind: ModuleKind::Metadata,
        object: Some(tmp_file),
        dwarf_object: None,
        bytecode: None,
        assembly: None,
        llvm_ir: None,
        links_from_incr_cache: Vec::new(),
    }
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

pub(crate) fn run_aot(
    tcx: TyCtxt<'_>,
    metadata: EncodedMetadata,
    need_metadata_module: bool,
) -> Box<OngoingCodegen> {
    // FIXME handle `-Ctarget-cpu=native`
    let target_cpu = match tcx.sess.opts.cg.target_cpu {
        Some(ref name) => name,
        None => tcx.sess.target.cpu.as_ref(),
    }
    .to_owned();

    let cgus = if tcx.sess.opts.output_types.should_codegen() {
        tcx.collect_and_partition_mono_items(()).codegen_units
    } else {
        // If only `--emit metadata` is used, we shouldn't perform any codegen.
        // Also `tcx.collect_and_partition_mono_items` may panic in that case.
        return Box::new(OngoingCodegen {
            modules: vec![],
            allocator_module: None,
            metadata_module: None,
            metadata,
            crate_info: CrateInfo::new(tcx, target_cpu),
            concurrency_limiter: ConcurrencyLimiter::new(0),
        });
    };

    if tcx.dep_graph.is_fully_enabled() {
        for cgu in cgus {
            tcx.ensure_ok().codegen_unit(cgu.name());
        }
    }

    // Calculate the CGU reuse
    let cgu_reuse = tcx.sess.time("find_cgu_reuse", || {
        cgus.iter().map(|cgu| determine_cgu_reuse(tcx, &cgu)).collect::<Vec<_>>()
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
        cgus.into_iter().enumerate().partition::<Vec<_>, _>(|&(i, _)| match cgu_reuse[i] {
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

    let metadata_module =
        if need_metadata_module { Some(emit_metadata_module(tcx, &metadata)) } else { None };

    Box::new(OngoingCodegen {
        modules,
        allocator_module,
        metadata_module,
        metadata,
        crate_info: CrateInfo::new(tcx, target_cpu),
        concurrency_limiter: concurrency_limiter.0,
    })
}
