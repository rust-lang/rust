//! The JIT driver uses [`cranelift_jit`] to JIT execute programs without writing any object
//! files.

use std::cell::RefCell;
use std::ffi::CString;
use std::lazy::SyncOnceCell;
use std::os::raw::{c_char, c_int};
use std::sync::{mpsc, Mutex};

use cranelift_codegen::binemit::{NullStackMapSink, NullTrapSink};
use rustc_codegen_ssa::CrateInfo;
use rustc_middle::mir::mono::MonoItem;
use rustc_session::Session;
use rustc_span::Symbol;

use cranelift_jit::{JITBuilder, JITModule};

use crate::{prelude::*, BackendConfig};
use crate::{CodegenCx, CodegenMode};

struct JitState {
    backend_config: BackendConfig,
    jit_module: JITModule,
}

thread_local! {
    static LAZY_JIT_STATE: RefCell<Option<JitState>> = const { RefCell::new(None) };
}

/// The Sender owned by the rustc thread
static GLOBAL_MESSAGE_SENDER: SyncOnceCell<Mutex<mpsc::Sender<UnsafeMessage>>> =
    SyncOnceCell::new();

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

fn create_jit_module<'tcx>(
    tcx: TyCtxt<'tcx>,
    backend_config: &BackendConfig,
    hotswap: bool,
) -> (JITModule, CodegenCx<'tcx>) {
    let crate_info = CrateInfo::new(tcx, "dummy_target_cpu".to_string());
    let imported_symbols = load_imported_symbols_for_jit(tcx.sess, crate_info);

    let isa = crate::build_isa(tcx.sess, backend_config);
    let mut jit_builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
    jit_builder.hotswap(hotswap);
    crate::compiler_builtins::register_functions_for_jit(&mut jit_builder);
    jit_builder.symbols(imported_symbols);
    let mut jit_module = JITModule::new(jit_builder);

    let mut cx = crate::CodegenCx::new(
        tcx,
        backend_config.clone(),
        jit_module.isa(),
        false,
        Symbol::intern("dummy_cgu_name"),
    );

    crate::allocator::codegen(tcx, &mut jit_module, &mut cx.unwind_context);
    crate::main_shim::maybe_create_entry_wrapper(
        tcx,
        &mut jit_module,
        &mut cx.unwind_context,
        true,
        true,
    );

    (jit_module, cx)
}

pub(crate) fn run_jit(tcx: TyCtxt<'_>, backend_config: BackendConfig) -> ! {
    if !tcx.sess.opts.output_types.should_codegen() {
        tcx.sess.fatal("JIT mode doesn't work with `cargo check`");
    }

    if !tcx.sess.crate_types().contains(&rustc_session::config::CrateType::Executable) {
        tcx.sess.fatal("can't jit non-executable crate");
    }

    let (mut jit_module, mut cx) = create_jit_module(
        tcx,
        &backend_config,
        matches!(backend_config.codegen_mode, CodegenMode::JitLazy),
    );

    let (_, cgus) = tcx.collect_and_partition_mono_items(());
    let mono_items = cgus
        .iter()
        .map(|cgu| cgu.items_in_deterministic_order(tcx).into_iter())
        .flatten()
        .collect::<FxHashMap<_, (_, _)>>()
        .into_iter()
        .collect::<Vec<(_, (_, _))>>();

    super::time(tcx, backend_config.display_cg_time, "codegen mono items", || {
        super::predefine_mono_items(tcx, &mut jit_module, &mono_items);
        for (mono_item, _) in mono_items {
            match mono_item {
                MonoItem::Fn(inst) => match backend_config.codegen_mode {
                    CodegenMode::Aot => unreachable!(),
                    CodegenMode::Jit => {
                        cx.tcx.sess.time("codegen fn", || {
                            crate::base::codegen_fn(&mut cx, &mut jit_module, inst)
                        });
                    }
                    CodegenMode::JitLazy => codegen_shim(&mut cx, &mut jit_module, inst),
                },
                MonoItem::Static(def_id) => {
                    crate::constant::codegen_static(tcx, &mut jit_module, def_id);
                }
                MonoItem::GlobalAsm(item_id) => {
                    let item = tcx.hir().item(item_id);
                    tcx.sess.span_fatal(item.span, "Global asm is not supported in JIT mode");
                }
            }
        }
    });

    if !cx.global_asm.is_empty() {
        tcx.sess.fatal("Inline asm is not supported in JIT mode");
    }

    tcx.sess.abort_if_errors();

    jit_module.finalize_definitions();
    unsafe { cx.unwind_context.register_jit(&jit_module) };

    println!(
        "Rustc codegen cranelift will JIT run the executable, because -Cllvm-args=mode=jit was passed"
    );

    let args = std::iter::once(&*tcx.crate_name(LOCAL_CRATE).as_str().to_string())
        .chain(backend_config.jit_args.iter().map(|arg| &**arg))
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

    LAZY_JIT_STATE.with(|lazy_jit_state| {
        let mut lazy_jit_state = lazy_jit_state.borrow_mut();
        assert!(lazy_jit_state.is_none());
        *lazy_jit_state = Some(JitState { backend_config, jit_module });
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

#[no_mangle]
extern "C" fn __clif_jit_fn(
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
            let backend_config = lazy_jit_state.backend_config.clone();

            let name = tcx.symbol_name(instance).name;
            let sig = crate::abi::get_function_sig(tcx, jit_module.isa().triple(), instance);
            let func_id = jit_module.declare_function(name, Linkage::Export, &sig).unwrap();

            let current_ptr = jit_module.read_got_entry(func_id);

            // If the function's GOT entry has already been updated to point at something other
            // than the shim trampoline, don't re-jit but just return the new pointer instead.
            // This does not need synchronization as this code is executed only by a sole rustc
            // thread.
            if current_ptr != trampoline_ptr {
                return current_ptr;
            }

            jit_module.prepare_for_function_redefine(func_id).unwrap();

            let mut cx = crate::CodegenCx::new(
                tcx,
                backend_config,
                jit_module.isa(),
                false,
                Symbol::intern("dummy_cgu_name"),
            );
            tcx.sess.time("codegen fn", || crate::base::codegen_fn(&mut cx, jit_module, instance));

            assert!(cx.global_asm.is_empty());
            jit_module.finalize_definitions();
            unsafe { cx.unwind_context.register_jit(&jit_module) };
            jit_module.get_finalized_function(func_id)
        })
    })
}

fn load_imported_symbols_for_jit(
    sess: &Session,
    crate_info: CrateInfo,
) -> Vec<(String, *const u8)> {
    use rustc_middle::middle::dependency_format::Linkage;

    let mut dylib_paths = Vec::new();

    let data = &crate_info
        .dependency_formats
        .iter()
        .find(|(crate_type, _data)| *crate_type == rustc_session::config::CrateType::Executable)
        .unwrap()
        .1;
    for &cnum in &crate_info.used_crates {
        let src = &crate_info.used_crate_source[&cnum];
        match data[cnum.as_usize() - 1] {
            Linkage::NotLinked | Linkage::IncludedFromDylib => {}
            Linkage::Static => {
                let name = &crate_info.crate_name[&cnum];
                let mut err = sess.struct_err(&format!("Can't load static lib {}", name.as_str()));
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
        let obj = object::File::parse(&*obj).unwrap();
        imported_symbols.extend(obj.dynamic_symbols().filter_map(|symbol| {
            let name = symbol.name().unwrap().to_string();
            if name.is_empty() || !symbol.is_global() || symbol.is_undefined() {
                return None;
            }
            if name.starts_with("rust_metadata_") {
                // The metadata is part of a section that is not loaded by the dynamic linker in
                // case of cg_llvm.
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

    sess.abort_if_errors();

    imported_symbols
}

fn codegen_shim<'tcx>(cx: &mut CodegenCx<'tcx>, module: &mut JITModule, inst: Instance<'tcx>) {
    let tcx = cx.tcx;

    let pointer_type = module.target_config().pointer_type();

    let name = tcx.symbol_name(inst).name;
    let sig = crate::abi::get_function_sig(tcx, module.isa().triple(), inst);
    let func_id = module.declare_function(name, Linkage::Export, &sig).unwrap();

    let instance_ptr = Box::into_raw(Box::new(inst));

    let jit_fn = module
        .declare_function(
            "__clif_jit_fn",
            Linkage::Import,
            &Signature {
                call_conv: module.target_config().default_call_conv,
                params: vec![AbiParam::new(pointer_type), AbiParam::new(pointer_type)],
                returns: vec![AbiParam::new(pointer_type)],
            },
        )
        .unwrap();

    cx.cached_context.clear();
    let trampoline = &mut cx.cached_context.func;
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

    module
        .define_function(
            func_id,
            &mut cx.cached_context,
            &mut NullTrapSink {},
            &mut NullStackMapSink {},
        )
        .unwrap();
}
