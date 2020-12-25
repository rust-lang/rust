//! The JIT driver uses [`cranelift_simplejit`] to JIT execute programs without writing any object
//! files.

use std::ffi::CString;
use std::os::raw::{c_char, c_int};

use rustc_codegen_ssa::CrateInfo;
use rustc_middle::mir::mono::MonoItem;

use cranelift_jit::{JITBuilder, JITModule};

use crate::prelude::*;

pub(super) fn run_jit(tcx: TyCtxt<'_>) -> ! {
    if !tcx.sess.opts.output_types.should_codegen() {
        tcx.sess.fatal("JIT mode doesn't work with `cargo check`.");
    }

    #[cfg(unix)]
    unsafe {
        // When not using our custom driver rustc will open us without the RTLD_GLOBAL flag, so
        // __cg_clif_global_atomic_mutex will not be exported. We fix this by opening ourself again
        // as global.
        // FIXME remove once atomic_shim is gone

        let mut dl_info: libc::Dl_info = std::mem::zeroed();
        assert_ne!(
            libc::dladdr(run_jit as *const libc::c_void, &mut dl_info),
            0
        );
        assert_ne!(
            libc::dlopen(dl_info.dli_fname, libc::RTLD_NOW | libc::RTLD_GLOBAL),
            std::ptr::null_mut(),
        );
    }

    let imported_symbols = load_imported_symbols_for_jit(tcx);

    let mut jit_builder = JITBuilder::with_isa(
        crate::build_isa(tcx.sess),
        cranelift_module::default_libcall_names(),
    );
    jit_builder.symbols(imported_symbols);
    let mut jit_module = JITModule::new(jit_builder);
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

    let mut cx = crate::CodegenCx::new(tcx, jit_module, false, false);

    super::time(tcx, "codegen mono items", || {
        super::predefine_mono_items(&mut cx, &mono_items);
        for (mono_item, (linkage, visibility)) in mono_items {
            let linkage = crate::linkage::get_clif_linkage(mono_item, linkage, visibility);
            match mono_item {
                MonoItem::Fn(inst) => {
                    cx.tcx.sess.time("codegen fn", || {
                        crate::base::codegen_fn(&mut cx, inst, linkage)
                    });
                }
                MonoItem::Static(def_id) => {
                    crate::constant::codegen_static(&mut cx.constants_cx, def_id)
                }
                MonoItem::GlobalAsm(hir_id) => {
                    let item = cx.tcx.hir().expect_item(hir_id);
                    tcx.sess
                        .span_fatal(item.span, "Global asm is not supported in JIT mode");
                }
            }
        }
    });

    let (mut jit_module, global_asm, _debug, mut unwind_context) =
        tcx.sess.time("finalize CodegenCx", || cx.finalize());
    jit_module.finalize_definitions();

    if !global_asm.is_empty() {
        tcx.sess.fatal("Inline asm is not supported in JIT mode");
    }

    crate::main_shim::maybe_create_entry_wrapper(tcx, &mut jit_module, &mut unwind_context, true);
    crate::allocator::codegen(tcx, &mut jit_module, &mut unwind_context);

    tcx.sess.abort_if_errors();

    jit_module.finalize_definitions();

    let _unwind_register_guard = unsafe { unwind_context.register_jit(&jit_module) };

    let finalized_main: *const u8 = jit_module.get_finalized_function(main_func_id);

    println!("Rustc codegen cranelift will JIT run the executable, because -Cllvm-args=mode=jit was passed");

    let f: extern "C" fn(c_int, *const *const c_char) -> c_int =
        unsafe { ::std::mem::transmute(finalized_main) };

    let args = ::std::env::var("CG_CLIF_JIT_ARGS").unwrap_or_else(|_| String::new());
    let args = std::iter::once(&*tcx.crate_name(LOCAL_CRATE).as_str().to_string())
        .chain(args.split(' '))
        .map(|arg| CString::new(arg).unwrap())
        .collect::<Vec<_>>();
    let mut argv = args.iter().map(|arg| arg.as_ptr()).collect::<Vec<_>>();

    // Push a null pointer as a terminating argument. This is required by POSIX and
    // useful as some dynamic linkers use it as a marker to jump over.
    argv.push(std::ptr::null());

    let ret = f(args.len() as c_int, argv.as_ptr());

    std::process::exit(ret);
}

fn load_imported_symbols_for_jit(tcx: TyCtxt<'_>) -> Vec<(String, *const u8)> {
    use rustc_middle::middle::dependency_format::Linkage;

    let mut dylib_paths = Vec::new();

    let crate_info = CrateInfo::new(tcx);
    let formats = tcx.dependency_formats(LOCAL_CRATE);
    let data = &formats
        .iter()
        .find(|(crate_type, _data)| *crate_type == rustc_session::config::CrateType::Executable)
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
        use object::{Object, ObjectSymbol};
        let lib = libloading::Library::new(&path).unwrap();
        let obj = std::fs::read(path).unwrap();
        let obj = object::File::parse(&obj).unwrap();
        imported_symbols.extend(obj.dynamic_symbols().filter_map(|symbol| {
            let name = symbol.name().unwrap().to_string();
            if name.is_empty() || !symbol.is_global() || symbol.is_undefined() {
                return None;
            }
            let dlsym_name = if cfg!(target_os = "macos") {
                // On macOS `dlsym` expects the name without leading `_`.
                assert!(name.starts_with('_'), "{:?}", name);
                &name[1..]
            } else {
                &name
            };
            let symbol: libloading::Symbol<'_, *const u8> =
                unsafe { lib.get(dlsym_name.as_bytes()) }.unwrap();
            Some((name, *symbol))
        }));
        std::mem::forget(lib)
    }

    tcx.sess.abort_if_errors();

    imported_symbols
}
