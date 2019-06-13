use std::any::Any;
use std::ffi::CString;
use std::fs::File;
use std::os::raw::{c_char, c_int};

use rustc::middle::cstore::EncodedMetadata;
use rustc::mir::mono::{Linkage as RLinkage, Visibility};
use rustc::session::config::{DebugInfo, OutputType};
use rustc_codegen_ssa::back::linker::LinkerInfo;
use rustc_codegen_ssa::CrateInfo;

use cranelift_faerie::*;

use crate::prelude::*;

pub fn codegen_crate<'a, 'tcx>(
    tcx: TyCtxt<'tcx, 'tcx>,
    metadata: EncodedMetadata,
    need_metadata_module: bool,
) -> Box<dyn Any> {
    if !tcx.sess.crate_types.get().contains(&CrateType::Executable)
        && std::env::var("SHOULD_RUN").is_ok()
    {
        tcx.sess
            .err("Can't JIT run non executable (SHOULD_RUN env var is set)");
    }

    tcx.sess.abort_if_errors();

    let mut log = if cfg!(debug_assertions) {
        Some(File::create(concat!(env!("CARGO_MANIFEST_DIR"), "/target/out/log.txt")).unwrap())
    } else {
        None
    };

    if std::env::var("SHOULD_RUN").is_ok() {
        #[cfg(not(target_arch = "wasm32"))]
        let _: ! = run_jit(tcx, &mut log);

        #[cfg(target_arch = "wasm32")]
        panic!("jit not supported on wasm");
    }

    run_aot(tcx, metadata, need_metadata_module, &mut log)
}

#[cfg(not(target_arch = "wasm32"))]
fn run_jit<'a, 'tcx: 'a>(tcx: TyCtxt<'tcx, 'tcx>, log: &mut Option<File>) -> ! {
    use cranelift_simplejit::{SimpleJITBackend, SimpleJITBuilder};

    let mut jit_module: Module<SimpleJITBackend> = Module::new(SimpleJITBuilder::new(
        cranelift_module::default_libcall_names(),
    ));
    assert_eq!(pointer_ty(tcx), jit_module.target_config().pointer_type());

    let sig = Signature {
        params: vec![
            AbiParam::new(jit_module.target_config().pointer_type()),
            AbiParam::new(jit_module.target_config().pointer_type()),
        ],
        returns: vec![AbiParam::new(
            jit_module.target_config().pointer_type(), /*isize*/
        )],
        call_conv: CallConv::SystemV,
    };
    let main_func_id = jit_module
        .declare_function("main", Linkage::Import, &sig)
        .unwrap();

    codegen_cgus(tcx, &mut jit_module, &mut None, log);
    crate::allocator::codegen(tcx.sess, &mut jit_module);
    jit_module.finalize_definitions();

    tcx.sess.abort_if_errors();

    let finalized_main: *const u8 = jit_module.get_finalized_function(main_func_id);

    println!("Rustc codegen cranelift will JIT run the executable, because the SHOULD_RUN env var is set");

    let f: extern "C" fn(c_int, *const *const c_char) -> c_int =
        unsafe { ::std::mem::transmute(finalized_main) };

    let args = ::std::env::var("JIT_ARGS").unwrap_or_else(|_| String::new());
    let args = args
        .split(" ")
        .chain(Some(&*tcx.crate_name(LOCAL_CRATE).as_str().to_string()))
        .map(|arg| CString::new(arg).unwrap())
        .collect::<Vec<_>>();
    let argv = args.iter().map(|arg| arg.as_ptr()).collect::<Vec<_>>();
    // TODO: Rust doesn't care, but POSIX argv has a NULL sentinel at the end

    let ret = f(args.len() as c_int, argv.as_ptr());

    jit_module.finish();
    std::process::exit(ret);
}

fn run_aot<'a, 'tcx: 'a>(
    tcx: TyCtxt<'tcx, 'tcx>,
    metadata: EncodedMetadata,
    need_metadata_module: bool,
    log: &mut Option<File>,
) -> Box<CodegenResults> {
    let new_module = |name: String| {
        let module: Module<FaerieBackend> = Module::new(
            FaerieBuilder::new(
                crate::build_isa(tcx.sess),
                name + ".o",
                FaerieTrapCollection::Disabled,
                cranelift_module::default_libcall_names(),
            )
            .unwrap(),
        );
        assert_eq!(pointer_ty(tcx), module.target_config().pointer_type());
        module
    };

    let emit_module = |name: &str,
                       kind: ModuleKind,
                       mut module: Module<FaerieBackend>,
                       debug: Option<DebugContext>| {
        module.finalize_definitions();
        let mut artifact = module.finish().artifact;

        if let Some(mut debug) = debug {
            debug.emit(&mut artifact);
        }

        let tmp_file = tcx
            .output_filenames(LOCAL_CRATE)
            .temp_path(OutputType::Object, Some(name));
        let obj = artifact.emit().unwrap();
        std::fs::write(&tmp_file, obj).unwrap();
        CompiledModule {
            name: name.to_string(),
            kind,
            object: Some(tmp_file),
            bytecode: None,
            bytecode_compressed: None,
        }
    };

    let mut faerie_module = new_module("some_file".to_string());

    let mut debug = if tcx.sess.opts.debuginfo != DebugInfo::None
        // macOS debuginfo doesn't work yet (see #303)
        && !tcx.sess.target.target.options.is_like_osx
    {
        let debug = DebugContext::new(
            tcx,
            faerie_module.target_config().pointer_type().bytes() as u8,
        );
        Some(debug)
    } else {
        None
    };

    codegen_cgus(tcx, &mut faerie_module, &mut debug, log);

    tcx.sess.abort_if_errors();

    let mut allocator_module = new_module("allocator_shim.o".to_string());
    let created_alloc_shim = crate::allocator::codegen(tcx.sess, &mut allocator_module);

    rustc_incremental::assert_dep_graph(tcx);
    rustc_incremental::save_dep_graph(tcx);
    rustc_incremental::finalize_session_directory(tcx.sess, tcx.crate_hash(LOCAL_CRATE));

    let metadata_module = if need_metadata_module {
        use rustc::mir::mono::CodegenUnitNameBuilder;

        let cgu_name_builder = &mut CodegenUnitNameBuilder::new(tcx);
        let metadata_cgu_name = cgu_name_builder
            .build_cgu_name(LOCAL_CRATE, &["crate"], Some("metadata"))
            .as_str()
            .to_string();

        let mut metadata_artifact =
            faerie::Artifact::new(crate::build_isa(tcx.sess).triple().clone(), metadata_cgu_name.clone());
        crate::metadata::write_metadata(tcx, &mut metadata_artifact);

        let tmp_file = tcx
            .output_filenames(LOCAL_CRATE)
            .temp_path(OutputType::Metadata, Some(&metadata_cgu_name));

        let obj = metadata_artifact.emit().unwrap();
        std::fs::write(&tmp_file, obj).unwrap();

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

    Box::new(CodegenResults {
        crate_name: tcx.crate_name(LOCAL_CRATE),
        modules: vec![emit_module(
            "dummy_name",
            ModuleKind::Regular,
            faerie_module,
            debug,
        )],
        allocator_module: if created_alloc_shim {
            Some(emit_module(
                "allocator_shim",
                ModuleKind::Allocator,
                allocator_module,
                None,
            ))
        } else {
            None
        },
        metadata_module,
        crate_hash: tcx.crate_hash(LOCAL_CRATE),
        metadata,
        windows_subsystem: None, // Windows is not yet supported
        linker_info: LinkerInfo::new(tcx),
        crate_info: CrateInfo::new(tcx),
    })
}

fn codegen_cgus<'a, 'tcx: 'a>(
    tcx: TyCtxt<'tcx, 'tcx>,
    module: &mut Module<impl Backend + 'static>,
    debug: &mut Option<DebugContext<'tcx>>,
    log: &mut Option<File>,
) {
    let (_, cgus) = tcx.collect_and_partition_mono_items(LOCAL_CRATE);
    let mono_items = cgus
        .iter()
        .map(|cgu| cgu.items_in_deterministic_order(tcx).into_iter())
        .flatten()
        .collect::<FxHashMap<_, (_, _)>>();

    codegen_mono_items(tcx, module, debug.as_mut(), log, mono_items);

    crate::main_shim::maybe_create_entry_wrapper(tcx, module);
}

fn codegen_mono_items<'a, 'tcx: 'a>(
    tcx: TyCtxt<'tcx, 'tcx>,
    module: &mut Module<impl Backend + 'static>,
    debug_context: Option<&mut DebugContext<'tcx>>,
    log: &mut Option<File>,
    mono_items: FxHashMap<MonoItem<'tcx>, (RLinkage, Visibility)>,
) {
    let mut cx = CodegenCx::new(tcx, module, debug_context);
    time("codegen mono items", move || {
        for (mono_item, (linkage, visibility)) in mono_items {
            crate::unimpl::try_unimpl(tcx, log, || {
                let linkage = crate::linkage::get_clif_linkage(mono_item, linkage, visibility);
                trans_mono_item(&mut cx, mono_item, linkage);
            });
        }

        cx.finalize();
    });
}

fn trans_mono_item<'a, 'clif, 'tcx: 'a, B: Backend + 'static>(
    cx: &mut crate::CodegenCx<'clif, 'tcx, B>,
    mono_item: MonoItem<'tcx>,
    linkage: Linkage,
) {
    let tcx = cx.tcx;
    match mono_item {
        MonoItem::Fn(inst) => {
            let _inst_guard =
                PrintOnPanic(|| format!("{:?} {}", inst, tcx.symbol_name(inst).as_str()));
            debug_assert!(!inst.substs.needs_infer());
            let _mir_guard = PrintOnPanic(|| {
                match inst.def {
                    InstanceDef::Item(_)
                    | InstanceDef::DropGlue(_, _)
                    | InstanceDef::Virtual(_, _) => {
                        let mut mir = ::std::io::Cursor::new(Vec::new());
                        crate::rustc_mir::util::write_mir_pretty(
                            tcx,
                            Some(inst.def_id()),
                            &mut mir,
                        )
                        .unwrap();
                        String::from_utf8(mir.into_inner()).unwrap()
                    }
                    _ => {
                        // FIXME fix write_mir_pretty for these instances
                        format!("{:#?}", tcx.instance_mir(inst.def))
                    }
                }
            });

            crate::base::trans_fn(cx, inst, linkage);
        }
        MonoItem::Static(def_id) => {
            crate::constant::codegen_static(&mut cx.ccx, def_id);
        }
        MonoItem::GlobalAsm(node_id) => tcx
            .sess
            .fatal(&format!("Unimplemented global asm mono item {:?}", node_id)),
    }
}

fn time<R>(name: &str, f: impl FnOnce() -> R) -> R {
    println!("[{}] start", name);
    let before = std::time::Instant::now();
    let res = f();
    let after = std::time::Instant::now();
    println!("[{}] end time: {:?}", name, after - before);
    res
}
