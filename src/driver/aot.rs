//! The AOT driver uses [`cranelift_object`] to write object files suitable for linking into a
//! standalone executable.

use std::path::PathBuf;

use rustc_codegen_ssa::back::linker::LinkerInfo;
use rustc_codegen_ssa::{CodegenResults, CompiledModule, CrateInfo, ModuleKind};
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_middle::dep_graph::{WorkProduct, WorkProductId};
use rustc_middle::middle::cstore::EncodedMetadata;
use rustc_middle::mir::mono::CodegenUnit;
use rustc_session::cgu_reuse_tracker::CguReuse;
use rustc_session::config::{DebugInfo, OutputType};

use cranelift_object::{ObjectModule, ObjectProduct};

use crate::prelude::*;

use crate::backend::AddConstructor;

fn new_module(tcx: TyCtxt<'_>, name: String) -> ObjectModule {
    let module = crate::backend::make_module(tcx.sess, name);
    assert_eq!(pointer_ty(tcx), module.target_config().pointer_type());
    module
}

struct ModuleCodegenResult(CompiledModule, Option<(WorkProductId, WorkProduct)>);

impl<HCX> HashStable<HCX> for ModuleCodegenResult {
    fn hash_stable(&self, _: &mut HCX, _: &mut StableHasher) {
        // do nothing
    }
}

fn emit_module(
    tcx: TyCtxt<'_>,
    name: String,
    kind: ModuleKind,
    module: ObjectModule,
    debug: Option<DebugContext<'_>>,
    unwind_context: UnwindContext<'_>,
    map_product: impl FnOnce(ObjectProduct) -> ObjectProduct,
) -> ModuleCodegenResult {
    let mut product = module.finish();

    if let Some(mut debug) = debug {
        debug.emit(&mut product);
    }

    unwind_context.emit(&mut product);

    let product = map_product(product);

    let tmp_file = tcx
        .output_filenames(LOCAL_CRATE)
        .temp_path(OutputType::Object, Some(&name));
    let obj = product.object.write().unwrap();
    if let Err(err) = std::fs::write(&tmp_file, obj) {
        tcx.sess
            .fatal(&format!("error writing object file: {}", err));
    }

    let work_product = if std::env::var("CG_CLIF_INCR_CACHE_DISABLED").is_ok() {
        None
    } else {
        rustc_incremental::copy_cgu_workproduct_to_incr_comp_cache_dir(
            tcx.sess,
            &name,
            &Some(tmp_file.clone()),
        )
    };

    ModuleCodegenResult(
        CompiledModule {
            name,
            kind,
            object: Some(tmp_file),
            bytecode: None,
        },
        work_product,
    )
}

fn reuse_workproduct_for_cgu(
    tcx: TyCtxt<'_>,
    cgu: &CodegenUnit<'_>,
    work_products: &mut FxHashMap<WorkProductId, WorkProduct>,
) -> CompiledModule {
    let incr_comp_session_dir = tcx.sess.incr_comp_session_dir();
    let mut object = None;
    let work_product = cgu.work_product(tcx);
    if let Some(saved_file) = &work_product.saved_file {
        let obj_out = tcx
            .output_filenames(LOCAL_CRATE)
            .temp_path(OutputType::Object, Some(&cgu.name().as_str()));
        object = Some(obj_out.clone());
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

    CompiledModule {
        name: cgu.name().to_string(),
        kind: ModuleKind::Regular,
        object,
        bytecode: None,
    }
}

fn module_codegen(tcx: TyCtxt<'_>, cgu_name: rustc_span::Symbol) -> ModuleCodegenResult {
    let cgu = tcx.codegen_unit(cgu_name);
    let mono_items = cgu.items_in_deterministic_order(tcx);

    let mut module = new_module(tcx, cgu_name.as_str().to_string());

    // Initialize the global atomic mutex using a constructor for proc-macros.
    // FIXME implement atomic instructions in Cranelift.
    let mut init_atomics_mutex_from_constructor = None;
    if tcx
        .sess
        .crate_types()
        .contains(&rustc_session::config::CrateType::ProcMacro)
    {
        if mono_items.iter().any(|(mono_item, _)| match mono_item {
            rustc_middle::mir::mono::MonoItem::Static(def_id) => tcx
                .symbol_name(Instance::mono(tcx, *def_id))
                .name
                .contains("__rustc_proc_macro_decls_"),
            _ => false,
        }) {
            init_atomics_mutex_from_constructor =
                Some(crate::atomic_shim::init_global_lock_constructor(
                    &mut module,
                    &format!("{}_init_atomics_mutex", cgu_name.as_str()),
                ));
        }
    }

    let mut cx = crate::CodegenCx::new(tcx, module, tcx.sess.opts.debuginfo != DebugInfo::None);
    super::predefine_mono_items(&mut cx, &mono_items);
    for (mono_item, (linkage, visibility)) in mono_items {
        let linkage = crate::linkage::get_clif_linkage(mono_item, linkage, visibility);
        super::codegen_mono_item(&mut cx, mono_item, linkage);
    }
    let (mut module, global_asm, debug, mut unwind_context) =
        tcx.sess.time("finalize CodegenCx", || cx.finalize());
    crate::main_shim::maybe_create_entry_wrapper(tcx, &mut module, &mut unwind_context, false);

    let codegen_result = emit_module(
        tcx,
        cgu.name().as_str().to_string(),
        ModuleKind::Regular,
        module,
        debug,
        unwind_context,
        |mut product| {
            if let Some(func_id) = init_atomics_mutex_from_constructor {
                product.add_constructor(func_id);
            }

            product
        },
    );

    codegen_global_asm(tcx, &cgu.name().as_str(), &global_asm);

    codegen_result
}

pub(super) fn run_aot(
    tcx: TyCtxt<'_>,
    metadata: EncodedMetadata,
    need_metadata_module: bool,
) -> Box<(CodegenResults, FxHashMap<WorkProductId, WorkProduct>)> {
    let mut work_products = FxHashMap::default();

    let cgus = if tcx.sess.opts.output_types.should_codegen() {
        tcx.collect_and_partition_mono_items(LOCAL_CRATE).1
    } else {
        // If only `--emit metadata` is used, we shouldn't perform any codegen.
        // Also `tcx.collect_and_partition_mono_items` may panic in that case.
        &[]
    };

    if tcx.dep_graph.is_fully_enabled() {
        for cgu in &*cgus {
            tcx.ensure().codegen_unit(cgu.name());
        }
    }

    let modules = super::time(tcx, "codegen mono items", || {
        cgus.iter()
            .map(|cgu| {
                let cgu_reuse = determine_cgu_reuse(tcx, cgu);
                tcx.sess
                    .cgu_reuse_tracker
                    .set_actual_reuse(&cgu.name().as_str(), cgu_reuse);

                match cgu_reuse {
                    _ if std::env::var("CG_CLIF_INCR_CACHE_DISABLED").is_ok() => {}
                    CguReuse::No => {}
                    CguReuse::PreLto => {
                        return reuse_workproduct_for_cgu(tcx, &*cgu, &mut work_products);
                    }
                    CguReuse::PostLto => unreachable!(),
                }

                let dep_node = cgu.codegen_dep_node(tcx);
                let (ModuleCodegenResult(module, work_product), _) = tcx.dep_graph.with_task(
                    dep_node,
                    tcx,
                    cgu.name(),
                    module_codegen,
                    rustc_middle::dep_graph::hash_result,
                );

                if let Some((id, product)) = work_product {
                    work_products.insert(id, product);
                }

                module
            })
            .collect::<Vec<_>>()
    });

    tcx.sess.abort_if_errors();

    let mut allocator_module = new_module(tcx, "allocator_shim".to_string());
    let mut allocator_unwind_context = UnwindContext::new(tcx, allocator_module.isa());
    let created_alloc_shim =
        crate::allocator::codegen(tcx, &mut allocator_module, &mut allocator_unwind_context);

    let allocator_module = if created_alloc_shim {
        let ModuleCodegenResult(module, work_product) = emit_module(
            tcx,
            "allocator_shim".to_string(),
            ModuleKind::Allocator,
            allocator_module,
            None,
            allocator_unwind_context,
            |product| product,
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
            use rustc_middle::mir::mono::CodegenUnitNameBuilder;

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

            if let Err(err) = std::fs::write(&tmp_file, obj) {
                tcx.sess
                    .fatal(&format!("error writing metadata object file: {}", err));
            }

            (metadata_cgu_name, tmp_file)
        });

        Some(CompiledModule {
            name: metadata_cgu_name,
            kind: ModuleKind::Metadata,
            object: Some(tmp_file),
            bytecode: None,
        })
    } else {
        None
    };

    if tcx.sess.opts.output_types.should_codegen() {
        rustc_incremental::assert_module_sources::assert_module_sources(tcx);
    }

    Box::new((
        CodegenResults {
            crate_name: tcx.crate_name(LOCAL_CRATE),
            modules,
            allocator_module,
            metadata_module,
            metadata,
            windows_subsystem: None, // Windows is not yet supported
            linker_info: LinkerInfo::new(tcx),
            crate_info: CrateInfo::new(tcx),
        },
        work_products,
    ))
}

fn codegen_global_asm(tcx: TyCtxt<'_>, cgu_name: &str, global_asm: &str) {
    use std::io::Write;
    use std::process::{Command, Stdio};

    if global_asm.is_empty() {
        return;
    }

    if cfg!(not(feature = "inline_asm"))
        || tcx.sess.target.is_like_osx
        || tcx.sess.target.is_like_windows
    {
        if global_asm.contains("__rust_probestack") {
            return;
        }

        // FIXME fix linker error on macOS
        if cfg!(not(feature = "inline_asm")) {
            tcx.sess.fatal(
                "asm! and global_asm! support is disabled while compiling rustc_codegen_cranelift",
            );
        } else {
            tcx.sess
                .fatal("asm! and global_asm! are not yet supported on macOS and Windows");
        }
    }

    let assembler = crate::toolchain::get_toolchain_binary(tcx.sess, "as");
    let linker = crate::toolchain::get_toolchain_binary(tcx.sess, "ld");

    // Remove all LLVM style comments
    let global_asm = global_asm
        .lines()
        .map(|line| {
            if let Some(index) = line.find("//") {
                &line[0..index]
            } else {
                line
            }
        })
        .collect::<Vec<_>>()
        .join("\n");

    let output_object_file = tcx
        .output_filenames(LOCAL_CRATE)
        .temp_path(OutputType::Object, Some(cgu_name));

    // Assemble `global_asm`
    let global_asm_object_file = add_file_stem_postfix(output_object_file.clone(), ".asm");
    let mut child = Command::new(assembler)
        .arg("-o")
        .arg(&global_asm_object_file)
        .stdin(Stdio::piped())
        .spawn()
        .expect("Failed to spawn `as`.");
    child
        .stdin
        .take()
        .unwrap()
        .write_all(global_asm.as_bytes())
        .unwrap();
    let status = child.wait().expect("Failed to wait for `as`.");
    if !status.success() {
        tcx.sess
            .fatal(&format!("Failed to assemble `{}`", global_asm));
    }

    // Link the global asm and main object file together
    let main_object_file = add_file_stem_postfix(output_object_file.clone(), ".main");
    std::fs::rename(&output_object_file, &main_object_file).unwrap();
    let status = Command::new(linker)
        .arg("-r") // Create a new object file
        .arg("-o")
        .arg(output_object_file)
        .arg(&main_object_file)
        .arg(&global_asm_object_file)
        .status()
        .unwrap();
    if !status.success() {
        tcx.sess.fatal(&format!(
            "Failed to link `{}` and `{}` together",
            main_object_file.display(),
            global_asm_object_file.display(),
        ));
    }

    std::fs::remove_file(global_asm_object_file).unwrap();
    std::fs::remove_file(main_object_file).unwrap();
}

fn add_file_stem_postfix(mut path: PathBuf, postfix: &str) -> PathBuf {
    let mut new_filename = path.file_stem().unwrap().to_owned();
    new_filename.push(postfix);
    if let Some(extension) = path.extension() {
        new_filename.push(".");
        new_filename.push(extension);
    }
    path.set_file_name(new_filename);
    path
}

// Adapted from https://github.com/rust-lang/rust/blob/303d8aff6092709edd4dbd35b1c88e9aa40bf6d8/src/librustc_codegen_ssa/base.rs#L922-L953
fn determine_cgu_reuse<'tcx>(tcx: TyCtxt<'tcx>, cgu: &CodegenUnit<'tcx>) -> CguReuse {
    if !tcx.dep_graph.is_fully_enabled() {
        return CguReuse::No;
    }

    let work_product_id = &cgu.work_product_id();
    if tcx
        .dep_graph
        .previous_work_product(work_product_id)
        .is_none()
    {
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
