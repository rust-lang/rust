//! The JIT driver uses [`cranelift_jit`] to JIT execute programs without writing any object
//! files.

use std::ffi::CString;
use std::os::raw::{c_char, c_int};

use cranelift_jit::{JITBuilder, JITModule};
use rustc_codegen_ssa::CrateInfo;
use rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrFlags;
use rustc_middle::mir::mono::MonoItem;
use rustc_session::Session;
use rustc_span::sym;

use crate::CodegenCx;
use crate::debuginfo::TypeDebugContext;
use crate::prelude::*;
use crate::unwind_module::UnwindModule;

fn create_jit_module(tcx: TyCtxt<'_>) -> (UnwindModule<JITModule>, CodegenCx) {
    let crate_info = CrateInfo::new(tcx, "dummy_target_cpu".to_string());

    let isa = crate::build_isa(tcx.sess, true);
    let mut jit_builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
    crate::compiler_builtins::register_functions_for_jit(&mut jit_builder);
    jit_builder.symbol_lookup_fn(dep_symbol_lookup_fn(tcx.sess, crate_info));
    let mut jit_module = UnwindModule::new(JITModule::new(jit_builder), false);

    let cx = crate::CodegenCx::new(tcx, jit_module.isa(), false, sym::dummy_cgu_name);

    crate::allocator::codegen(tcx, &mut jit_module);

    (jit_module, cx)
}

pub(crate) fn run_jit(tcx: TyCtxt<'_>, jit_args: Vec<String>) -> ! {
    if !tcx.sess.opts.output_types.should_codegen() {
        tcx.dcx().fatal("JIT mode doesn't work with `cargo check`");
    }

    if !tcx.crate_types().contains(&rustc_session::config::CrateType::Executable) {
        tcx.dcx().fatal("can't jit non-executable crate");
    }

    let (mut jit_module, mut cx) = create_jit_module(tcx);
    let mut cached_context = Context::new();

    let cgus = tcx.collect_and_partition_mono_items(()).codegen_units;
    let mono_items = cgus
        .iter()
        .map(|cgu| cgu.items_in_deterministic_order(tcx).into_iter())
        .flatten()
        .collect::<FxHashMap<_, _>>()
        .into_iter()
        .collect::<Vec<(_, _)>>();

    tcx.sess.time("codegen mono items", || {
        super::predefine_mono_items(tcx, &mut jit_module, &mono_items);
        for (mono_item, _) in mono_items {
            match mono_item {
                MonoItem::Fn(inst) => {
                    codegen_and_compile_fn(
                        tcx,
                        &mut cx,
                        &mut cached_context,
                        &mut jit_module,
                        inst,
                    );
                }
                MonoItem::Static(def_id) => {
                    crate::constant::codegen_static(tcx, &mut jit_module, def_id);
                }
                MonoItem::GlobalAsm(item_id) => {
                    let item = tcx.hir_item(item_id);
                    tcx.dcx().span_fatal(item.span, "Global asm is not supported in JIT mode");
                }
            }
        }
    });

    if !cx.global_asm.is_empty() {
        tcx.dcx().fatal("Inline asm is not supported in JIT mode");
    }

    crate::main_shim::maybe_create_entry_wrapper(tcx, &mut jit_module, true, true);

    tcx.dcx().abort_if_errors();

    let mut jit_module = jit_module.finalize_definitions();

    println!(
        "Rustc codegen cranelift will JIT run the executable, because -Cllvm-args=mode=jit was passed"
    );

    let args = std::iter::once(&*tcx.crate_name(LOCAL_CRATE).as_str().to_string())
        .chain(jit_args.iter().map(|arg| &**arg))
        .map(|arg| CString::new(arg).unwrap())
        .collect::<Vec<_>>();

    let start_sig = Signature {
        params: vec![
            AbiParam::new(jit_module.target_config().pointer_type()),
            AbiParam::new(jit_module.target_config().pointer_type()),
        ],
        returns: vec![AbiParam::new(jit_module.target_config().pointer_type() /*isize*/)],
        call_conv: jit_module.target_config().default_call_conv,
    };
    let start_func_id = jit_module.declare_function("main", Linkage::Import, &start_sig).unwrap();
    let finalized_start: *const u8 = jit_module.get_finalized_function(start_func_id);

    let f: extern "C" fn(c_int, *const *const c_char) -> c_int =
        unsafe { ::std::mem::transmute(finalized_start) };

    let mut argv = args.iter().map(|arg| arg.as_ptr()).collect::<Vec<_>>();

    // Push a null pointer as a terminating argument. This is required by POSIX and
    // useful as some dynamic linkers use it as a marker to jump over.
    argv.push(std::ptr::null());

    let ret = f(args.len() as c_int, argv.as_ptr());
    std::process::exit(ret);
}

fn codegen_and_compile_fn<'tcx>(
    tcx: TyCtxt<'tcx>,
    cx: &mut crate::CodegenCx,
    cached_context: &mut Context,
    module: &mut dyn Module,
    instance: Instance<'tcx>,
) {
    if tcx.codegen_instance_attrs(instance.def).flags.contains(CodegenFnAttrFlags::NAKED) {
        tcx.dcx()
            .span_fatal(tcx.def_span(instance.def_id()), "Naked asm is not supported in JIT mode");
    }

    cranelift_codegen::timing::set_thread_profiler(Box::new(super::MeasuremeProfiler(
        tcx.prof.clone(),
    )));

    tcx.prof.generic_activity("codegen and compile fn").run(|| {
        let _inst_guard =
            crate::PrintOnPanic(|| format!("{:?} {}", instance, tcx.symbol_name(instance).name));

        let cached_func = std::mem::replace(&mut cached_context.func, Function::new());
        let codegened_func = crate::base::codegen_fn(
            tcx,
            cx,
            &mut TypeDebugContext::default(),
            cached_func,
            module,
            instance,
        );
        crate::base::compile_fn(cx, &tcx.prof, cached_context, module, codegened_func);
    });
}

fn dep_symbol_lookup_fn(
    sess: &Session,
    crate_info: CrateInfo,
) -> Box<dyn Fn(&str) -> Option<*const u8> + Send> {
    use rustc_middle::middle::dependency_format::Linkage;

    let mut dylib_paths = Vec::new();

    let data = &crate_info.dependency_formats[&rustc_session::config::CrateType::Executable];
    // `used_crates` is in reverse postorder in terms of dependencies. Reverse the order here to
    // get a postorder which ensures that all dependencies of a dylib are loaded before the dylib
    // itself. This helps the dynamic linker to find dylibs not in the regular dynamic library
    // search path.
    for &cnum in crate_info.used_crates.iter().rev() {
        let src = &crate_info.used_crate_source[&cnum];
        match data[cnum] {
            Linkage::NotLinked | Linkage::IncludedFromDylib => {}
            Linkage::Static => {
                let name = crate_info.crate_name[&cnum];
                let mut diag = sess.dcx().struct_err(format!("Can't load static lib {}", name));
                diag.note("rustc_codegen_cranelift can only load dylibs in JIT mode.");
                diag.emit();
            }
            Linkage::Dynamic => {
                dylib_paths.push(src.dylib.as_ref().unwrap().0.clone());
            }
        }
    }

    let imported_dylibs = Box::leak(
        dylib_paths
            .into_iter()
            .map(|path| unsafe { libloading::Library::new(&path).unwrap() })
            .collect::<Box<[_]>>(),
    );

    sess.dcx().abort_if_errors();

    Box::new(move |sym_name| {
        for dylib in &*imported_dylibs {
            if let Ok(sym) = unsafe { dylib.get::<*const u8>(sym_name.as_bytes()) } {
                return Some(*sym);
            }
        }
        None
    })
}
