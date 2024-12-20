//! The JIT driver uses [`cranelift_jit`] to JIT execute programs without writing any object
//! files.

use std::cell::RefCell;
use std::ffi::CString;
use std::os::raw::{c_char, c_int};
use std::sync::{Mutex, OnceLock, mpsc};

use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use rustc_codegen_ssa::CrateInfo;
use rustc_middle::mir::mono::MonoItem;
use rustc_session::Session;
use rustc_span::sym;

use crate::debuginfo::TypeDebugContext;
use crate::prelude::*;
use crate::unwind_module::UnwindModule;
use crate::{CodegenCx, CodegenMode};

struct JitState {
    jit_module: UnwindModule<JITModule>,
}

thread_local! {
    static LAZY_JIT_STATE: RefCell<Option<JitState>> = const { RefCell::new(None) };
}

/// The Sender owned by the rustc thread
static GLOBAL_MESSAGE_SENDER: OnceLock<Mutex<mpsc::Sender<UnsafeMessage>>> = OnceLock::new();

/// A message that is sent from the jitted runtime to the rustc thread.
/// Senders are responsible for upholding `Send` semantics.
enum UnsafeMessage {
    /// Request that the specified `Instance` be lazily jitted.
    ///
    /// Nothing accessible through `instance_ptr` may be moved or mutated by the sender after
    /// this message is sent.
    JitFn {
        instance_ptr: *const Instance<'static>,
        trampoline_ptr: *const u8,
        tx: mpsc::Sender<*const u8>,
    },
}
unsafe impl Send for UnsafeMessage {}

impl UnsafeMessage {
    /// Send the message.
    fn send(self) -> Result<(), mpsc::SendError<UnsafeMessage>> {
        thread_local! {
            /// The Sender owned by the local thread
            static LOCAL_MESSAGE_SENDER: mpsc::Sender<UnsafeMessage> =
                GLOBAL_MESSAGE_SENDER
                    .get().unwrap()
                    .lock().unwrap()
                    .clone();
        }
        LOCAL_MESSAGE_SENDER.with(|sender| sender.send(self))
    }
}

fn create_jit_module(tcx: TyCtxt<'_>, hotswap: bool) -> (UnwindModule<JITModule>, CodegenCx) {
    let crate_info = CrateInfo::new(tcx, "dummy_target_cpu".to_string());

    let isa = crate::build_isa(tcx.sess);
    let mut jit_builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
    jit_builder.hotswap(hotswap);
    crate::compiler_builtins::register_functions_for_jit(&mut jit_builder);
    jit_builder.symbol_lookup_fn(dep_symbol_lookup_fn(tcx.sess, crate_info));
    jit_builder.symbol("__clif_jit_fn", clif_jit_fn as *const u8);
    let mut jit_module = UnwindModule::new(JITModule::new(jit_builder), false);

    let cx = crate::CodegenCx::new(tcx, jit_module.isa(), false, sym::dummy_cgu_name);

    crate::allocator::codegen(tcx, &mut jit_module);

    (jit_module, cx)
}

pub(crate) fn run_jit(tcx: TyCtxt<'_>, codegen_mode: CodegenMode, jit_args: Vec<String>) -> ! {
    if !tcx.sess.opts.output_types.should_codegen() {
        tcx.dcx().fatal("JIT mode doesn't work with `cargo check`");
    }

    if !tcx.crate_types().contains(&rustc_session::config::CrateType::Executable) {
        tcx.dcx().fatal("can't jit non-executable crate");
    }

    let (mut jit_module, mut cx) =
        create_jit_module(tcx, matches!(codegen_mode, CodegenMode::JitLazy));
    let mut cached_context = Context::new();

    let (_, cgus) = tcx.collect_and_partition_mono_items(());
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
                MonoItem::Fn(inst) => match codegen_mode {
                    CodegenMode::Aot => unreachable!(),
                    CodegenMode::Jit => {
                        codegen_and_compile_fn(
                            tcx,
                            &mut cx,
                            &mut cached_context,
                            &mut jit_module,
                            inst,
                        );
                    }
                    CodegenMode::JitLazy => {
                        codegen_shim(tcx, &mut cached_context, &mut jit_module, inst)
                    }
                },
                MonoItem::Static(def_id) => {
                    crate::constant::codegen_static(tcx, &mut jit_module, def_id);
                }
                MonoItem::GlobalAsm(item_id) => {
                    let item = tcx.hir().item(item_id);
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

    jit_module.finalize_definitions();

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
    let finalized_start: *const u8 = jit_module.module.get_finalized_function(start_func_id);

    LAZY_JIT_STATE.with(|lazy_jit_state| {
        let mut lazy_jit_state = lazy_jit_state.borrow_mut();
        assert!(lazy_jit_state.is_none());
        *lazy_jit_state = Some(JitState { jit_module });
    });

    let f: extern "C" fn(c_int, *const *const c_char) -> c_int =
        unsafe { ::std::mem::transmute(finalized_start) };

    let (tx, rx) = mpsc::channel();
    GLOBAL_MESSAGE_SENDER.set(Mutex::new(tx)).unwrap();

    // Spawn the jitted runtime in a new thread so that this rustc thread can handle messages
    // (eg to lazily JIT further functions as required)
    std::thread::spawn(move || {
        let mut argv = args.iter().map(|arg| arg.as_ptr()).collect::<Vec<_>>();

        // Push a null pointer as a terminating argument. This is required by POSIX and
        // useful as some dynamic linkers use it as a marker to jump over.
        argv.push(std::ptr::null());

        let ret = f(args.len() as c_int, argv.as_ptr());
        std::process::exit(ret);
    });

    // Handle messages
    loop {
        match rx.recv().unwrap() {
            // lazy JIT compilation request - compile requested instance and return pointer to result
            UnsafeMessage::JitFn { instance_ptr, trampoline_ptr, tx } => {
                tx.send(jit_fn(instance_ptr, trampoline_ptr))
                    .expect("jitted runtime hung up before response to lazy JIT request was sent");
            }
        }
    }
}

pub(crate) fn codegen_and_compile_fn<'tcx>(
    tcx: TyCtxt<'tcx>,
    cx: &mut crate::CodegenCx,
    cached_context: &mut Context,
    module: &mut dyn Module,
    instance: Instance<'tcx>,
) {
    cranelift_codegen::timing::set_thread_profiler(Box::new(super::MeasuremeProfiler(
        tcx.prof.clone(),
    )));

    tcx.prof.generic_activity("codegen and compile fn").run(|| {
        let _inst_guard =
            crate::PrintOnPanic(|| format!("{:?} {}", instance, tcx.symbol_name(instance).name));

        let cached_func = std::mem::replace(&mut cached_context.func, Function::new());
        if let Some(codegened_func) = crate::base::codegen_fn(
            tcx,
            cx,
            &mut TypeDebugContext::default(),
            cached_func,
            module,
            instance,
        ) {
            crate::base::compile_fn(cx, &tcx.prof, cached_context, module, codegened_func);
        }
    });
}

extern "C" fn clif_jit_fn(
    instance_ptr: *const Instance<'static>,
    trampoline_ptr: *const u8,
) -> *const u8 {
    // send the JIT request to the rustc thread, with a channel for the response
    let (tx, rx) = mpsc::channel();
    UnsafeMessage::JitFn { instance_ptr, trampoline_ptr, tx }
        .send()
        .expect("rustc thread hung up before lazy JIT request was sent");

    // block on JIT compilation result
    rx.recv().expect("rustc thread hung up before responding to sent lazy JIT request")
}

fn jit_fn(instance_ptr: *const Instance<'static>, trampoline_ptr: *const u8) -> *const u8 {
    rustc_middle::ty::tls::with(|tcx| {
        // lift is used to ensure the correct lifetime for instance.
        let instance = tcx.lift(unsafe { *instance_ptr }).unwrap();

        LAZY_JIT_STATE.with(|lazy_jit_state| {
            let mut lazy_jit_state = lazy_jit_state.borrow_mut();
            let lazy_jit_state = lazy_jit_state.as_mut().unwrap();
            let jit_module = &mut lazy_jit_state.jit_module;

            let name = tcx.symbol_name(instance).name;
            let sig = crate::abi::get_function_sig(
                tcx,
                jit_module.target_config().default_call_conv,
                instance,
            );
            let func_id = jit_module.declare_function(name, Linkage::Export, &sig).unwrap();

            let current_ptr = jit_module.module.read_got_entry(func_id);

            // If the function's GOT entry has already been updated to point at something other
            // than the shim trampoline, don't re-jit but just return the new pointer instead.
            // This does not need synchronization as this code is executed only by a sole rustc
            // thread.
            if current_ptr != trampoline_ptr {
                return current_ptr;
            }

            jit_module.module.prepare_for_function_redefine(func_id).unwrap();

            let mut cx = crate::CodegenCx::new(tcx, jit_module.isa(), false, sym::dummy_cgu_name);
            codegen_and_compile_fn(tcx, &mut cx, &mut Context::new(), jit_module, instance);

            assert!(cx.global_asm.is_empty());
            jit_module.finalize_definitions();
            jit_module.module.get_finalized_function(func_id)
        })
    })
}

fn dep_symbol_lookup_fn(
    sess: &Session,
    crate_info: CrateInfo,
) -> Box<dyn Fn(&str) -> Option<*const u8> + Send> {
    use rustc_middle::middle::dependency_format::Linkage;

    let mut dylib_paths = Vec::new();

    let data = &crate_info.dependency_formats[&rustc_session::config::CrateType::Executable].1;
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

fn codegen_shim<'tcx>(
    tcx: TyCtxt<'tcx>,
    cached_context: &mut Context,
    module: &mut UnwindModule<JITModule>,
    inst: Instance<'tcx>,
) {
    let pointer_type = module.target_config().pointer_type();

    let name = tcx.symbol_name(inst).name;
    let sig = crate::abi::get_function_sig(tcx, module.target_config().default_call_conv, inst);
    let func_id = module.declare_function(name, Linkage::Export, &sig).unwrap();

    let instance_ptr = Box::into_raw(Box::new(inst));

    let jit_fn = module
        .declare_function("__clif_jit_fn", Linkage::Import, &Signature {
            call_conv: module.target_config().default_call_conv,
            params: vec![AbiParam::new(pointer_type), AbiParam::new(pointer_type)],
            returns: vec![AbiParam::new(pointer_type)],
        })
        .unwrap();

    let context = cached_context;
    context.clear();
    let trampoline = &mut context.func;
    trampoline.signature = sig.clone();

    let mut builder_ctx = FunctionBuilderContext::new();
    let mut trampoline_builder = FunctionBuilder::new(trampoline, &mut builder_ctx);

    let trampoline_fn = module.declare_func_in_func(func_id, trampoline_builder.func);
    let jit_fn = module.declare_func_in_func(jit_fn, trampoline_builder.func);
    let sig_ref = trampoline_builder.func.import_signature(sig);

    let entry_block = trampoline_builder.create_block();
    trampoline_builder.append_block_params_for_function_params(entry_block);
    let fn_args = trampoline_builder.func.dfg.block_params(entry_block).to_vec();

    trampoline_builder.switch_to_block(entry_block);
    let instance_ptr = trampoline_builder.ins().iconst(pointer_type, instance_ptr as u64 as i64);
    let trampoline_ptr = trampoline_builder.ins().func_addr(pointer_type, trampoline_fn);
    let jitted_fn = trampoline_builder.ins().call(jit_fn, &[instance_ptr, trampoline_ptr]);
    let jitted_fn = trampoline_builder.func.dfg.inst_results(jitted_fn)[0];
    let call_inst = trampoline_builder.ins().call_indirect(sig_ref, jitted_fn, &fn_args);
    let ret_vals = trampoline_builder.func.dfg.inst_results(call_inst).to_vec();
    trampoline_builder.ins().return_(&ret_vals);

    module.define_function(func_id, context).unwrap();
}
