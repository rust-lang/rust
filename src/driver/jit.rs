use std::ffi::CString;
use std::os::raw::{c_char, c_int};

use rustc_codegen_ssa::CrateInfo;

use crate::prelude::*;

pub(super) fn run_jit(tcx: TyCtxt<'_>) -> ! {
    use cranelift_simplejit::{SimpleJITBackend, SimpleJITBuilder};

    // Rustc opens us without the RTLD_GLOBAL flag, so __cg_clif_global_atomic_mutex will not be
    // exported. We fix this by opening ourself again as global.
    // FIXME remove once atomic_shim is gone
    let cg_dylib = std::ffi::OsString::from(&tcx.sess.opts.debugging_opts.codegen_backend.as_ref().unwrap());
    std::mem::forget(libloading::os::unix::Library::open(Some(cg_dylib), libc::RTLD_NOW | libc::RTLD_GLOBAL).unwrap());


    let imported_symbols = load_imported_symbols_for_jit(tcx);

    let mut jit_builder = SimpleJITBuilder::with_isa(
        crate::build_isa(tcx.sess, false),
        cranelift_module::default_libcall_names(),
    );
    jit_builder.symbols(imported_symbols);
    let mut jit_module: Module<SimpleJITBackend> = Module::new(jit_builder);
    assert_eq!(pointer_ty(tcx), jit_module.target_config().pointer_type());

    let sig = Signature {
        params: vec![
            AbiParam::new(jit_module.target_config().pointer_type()),
            AbiParam::new(jit_module.target_config().pointer_type()),
        ],
        returns: vec![AbiParam::new(
            jit_module.target_config().pointer_type(), /*isize*/
        )],
        call_conv: CallConv::triple_default(&crate::target_triple(tcx.sess)),
    };
    let main_func_id = jit_module
        .declare_function("main", Linkage::Import, &sig)
        .unwrap();

    let (_, cgus) = tcx.collect_and_partition_mono_items(LOCAL_CRATE);
    let mono_items = cgus
        .iter()
        .map(|cgu| cgu.items_in_deterministic_order(tcx).into_iter())
        .flatten()
        .collect::<FxHashMap<_, (_, _)>>()
        .into_iter()
        .collect::<Vec<(_, (_, _))>>();

    super::time(tcx, "codegen mono items", || {
        super::codegen_mono_items(tcx, &mut jit_module, None, mono_items);
    });
    crate::main_shim::maybe_create_entry_wrapper(tcx, &mut jit_module);
    crate::allocator::codegen(tcx, &mut jit_module);

    jit_module.finalize_definitions();

    tcx.sess.abort_if_errors();

    let finalized_main: *const u8 = jit_module.get_finalized_function(main_func_id);

    println!("Rustc codegen cranelift will JIT run the executable, because the CG_CLIF_JIT env var is set");

    let f: extern "C" fn(c_int, *const *const c_char) -> c_int =
        unsafe { ::std::mem::transmute(finalized_main) };

    let args = ::std::env::var("CG_CLIF_JIT_ARGS").unwrap_or_else(|_| String::new());
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

fn load_imported_symbols_for_jit(tcx: TyCtxt<'_>) -> Vec<(String, *const u8)> {
    use rustc::middle::dependency_format::Linkage;

    let mut dylib_paths = Vec::new();

    let crate_info = CrateInfo::new(tcx);
    let formats = tcx.dependency_formats(LOCAL_CRATE);
    let data = &formats
        .iter()
        .find(|(crate_type, _data)| *crate_type == CrateType::Executable)
        .unwrap()
        .1;
    for &(cnum, _) in &crate_info.used_crates_dynamic {
        let src = &crate_info.used_crate_source[&cnum];
        match data[cnum.as_usize() - 1] {
            Linkage::NotLinked | Linkage::IncludedFromDylib => {}
            Linkage::Static => {
                let name = tcx.crate_name(cnum);
                let mut err = tcx
                    .sess
                    .struct_err(&format!("Can't load static lib {}", name.as_str()));
                err.note("rustc_codegen_cranelift can only load dylibs in JIT mode.");
                err.emit();
            }
            Linkage::Dynamic => {
                dylib_paths.push(src.dylib.as_ref().unwrap().0.clone());
            }
        }
    }

    let mut imported_symbols = Vec::new();
    for path in dylib_paths {
        use object::Object;
        let lib = libloading::Library::new(&path).unwrap();
        let obj = std::fs::read(path).unwrap();
        let obj = object::File::parse(&obj).unwrap();
        imported_symbols.extend(obj.dynamic_symbols().filter_map(|(_idx, symbol)| {
            let name = symbol.name().unwrap().to_string();
            if name.is_empty() || !symbol.is_global() || symbol.is_undefined() {
                return None;
            }
            let dlsym_name = if cfg!(target_os = "macos") {
                // On macOS `dlsym` expects the name without leading `_`.
                assert!(name.starts_with("_"), "{:?}", name);
                &name[1..]
            } else {
                &name
            };
            let symbol: libloading::Symbol<*const u8> =
                unsafe { lib.get(dlsym_name.as_bytes()) }.unwrap();
            Some((name, *symbol))
        }));
        std::mem::forget(lib)
    }

    tcx.sess.abort_if_errors();

    imported_symbols
}
