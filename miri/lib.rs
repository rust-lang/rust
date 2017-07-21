#![feature(
    i128_type,
    rustc_private,
)]

// From rustc.
#[macro_use]
extern crate log;
extern crate log_settings;
extern crate rustc;
extern crate rustc_const_math;
extern crate rustc_data_structures;
extern crate syntax;

use rustc::ty::{self, TyCtxt, Ty};
use rustc::hir::def_id::{DefId, CRATE_DEF_INDEX};
use rustc::mir;
use syntax::attr;
use syntax::abi::Abi;

use std::mem;
use std::collections::{
    HashMap,
    BTreeMap,
};

extern crate rustc_miri;
pub use rustc_miri::interpret::*;

pub fn eval_main<'a, 'tcx: 'a>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    main_id: DefId,
    start_wrapper: Option<DefId>,
    limits: ResourceLimits,
) {
    fn run_main<'a, 'tcx: 'a>(
        ecx: &mut rustc_miri::interpret::EvalContext<'a, 'tcx, Evaluator>,
        main_id: DefId,
        start_wrapper: Option<DefId>,
    ) -> EvalResult<'tcx> {
        let main_instance = ty::Instance::mono(ecx.tcx, main_id);
        let main_mir = ecx.load_mir(main_instance.def)?;
        let mut cleanup_ptr = None; // Pointer to be deallocated when we are done

        if !main_mir.return_ty.is_nil() || main_mir.arg_count != 0 {
            return Err(EvalError::Unimplemented("miri does not support main functions without `fn()` type signatures".to_owned()));
        }

        if let Some(start_id) = start_wrapper {
            let start_instance = ty::Instance::mono(ecx.tcx, start_id);
            let start_mir = ecx.load_mir(start_instance.def)?;

            if start_mir.arg_count != 3 {
                return Err(EvalError::AbiViolation(format!("'start' lang item should have three arguments, but has {}", start_mir.arg_count)));
            }

            // Return value
            let size = ecx.tcx.data_layout.pointer_size.bytes();
            let align = ecx.tcx.data_layout.pointer_align.abi();
            let ret_ptr = ecx.memory_mut().allocate(size, align, Kind::Stack)?;
            cleanup_ptr = Some(ret_ptr);

            // Push our stack frame
            ecx.push_stack_frame(
                start_instance,
                start_mir.span,
                start_mir,
                Lvalue::from_ptr(ret_ptr),
                StackPopCleanup::None,
            )?;

            let mut args = ecx.frame().mir.args_iter();

            // First argument: pointer to main()
            let main_ptr = ecx.memory_mut().create_fn_alloc(main_instance);
            let dest = ecx.eval_lvalue(&mir::Lvalue::Local(args.next().unwrap()))?;
            let main_ty = main_instance.def.def_ty(ecx.tcx);
            let main_ptr_ty = ecx.tcx.mk_fn_ptr(main_ty.fn_sig(ecx.tcx));
            ecx.write_value(Value::ByVal(PrimVal::Ptr(main_ptr)), dest, main_ptr_ty)?;

            // Second argument (argc): 0
            let dest = ecx.eval_lvalue(&mir::Lvalue::Local(args.next().unwrap()))?;
            let ty = ecx.tcx.types.isize;
            ecx.write_null(dest, ty)?;

            // Third argument (argv): 0
            let dest = ecx.eval_lvalue(&mir::Lvalue::Local(args.next().unwrap()))?;
            let ty = ecx.tcx.mk_imm_ptr(ecx.tcx.mk_imm_ptr(ecx.tcx.types.u8));
            ecx.write_null(dest, ty)?;
        } else {
            ecx.push_stack_frame(
                main_instance,
                main_mir.span,
                main_mir,
                Lvalue::undef(),
                StackPopCleanup::None,
            )?;
        }

        while ecx.step()? {}
        ecx.finish()?;
        if let Some(cleanup_ptr) = cleanup_ptr {
            ecx.memory_mut().deallocate(cleanup_ptr, None, Kind::Stack)?;
        }
        Ok(())
    }

    let mut ecx = EvalContext::new(tcx, limits, Default::default(), Default::default());
    match run_main(&mut ecx, main_id, start_wrapper) {
        Ok(()) => {
            let leaks = ecx.memory().leak_report();
            if leaks != 0 {
                tcx.sess.err("the evaluated program leaked memory");
            }
        }
        Err(e) => {
            ecx.report(&e);
        }
    }
}

struct Evaluator;
#[derive(Default)]
struct EvaluatorData {
    /// Environment variables set by `setenv`
    /// Miri does not expose env vars from the host to the emulated program
    pub(crate) env_vars: HashMap<Vec<u8>, MemoryPointer>,
}

pub type TlsKey = usize;

#[derive(Copy, Clone, Debug)]
pub struct TlsEntry<'tcx> {
    data: Pointer, // Will eventually become a map from thread IDs to `Pointer`s, if we ever support more than one thread.
    dtor: Option<ty::Instance<'tcx>>,
}

#[derive(Default)]
struct MemoryData<'tcx> {
    /// The Key to use for the next thread-local allocation.
    next_thread_local: TlsKey,

    /// pthreads-style thread-local storage.
    thread_local: BTreeMap<TlsKey, TlsEntry<'tcx>>,
}

trait EvalContextExt<'tcx> {
    fn call_c_abi(
        &mut self,
        def_id: DefId,
        arg_operands: &[mir::Operand<'tcx>],
        dest: Lvalue<'tcx>,
        dest_ty: Ty<'tcx>,
        dest_block: mir::BasicBlock,
    ) -> EvalResult<'tcx>;
    fn finish(&mut self) -> EvalResult<'tcx>;

    fn resolve_path(&self, path: &[&str]) -> EvalResult<'tcx, ty::Instance<'tcx>>;

    fn call_missing_fn(
        &mut self,
        instance: ty::Instance<'tcx>,
        destination: Option<(Lvalue<'tcx>, mir::BasicBlock)>,
        arg_operands: &[mir::Operand<'tcx>],
        sig: ty::FnSig<'tcx>,
        path: String,
    ) -> EvalResult<'tcx>;
}

impl<'a, 'tcx> EvalContextExt<'tcx> for EvalContext<'a, 'tcx, Evaluator> {
    fn call_c_abi(
        &mut self,
        def_id: DefId,
        arg_operands: &[mir::Operand<'tcx>],
        dest: Lvalue<'tcx>,
        dest_ty: Ty<'tcx>,
        dest_block: mir::BasicBlock,
    ) -> EvalResult<'tcx> {
        let name = self.tcx.item_name(def_id);
        let attrs = self.tcx.get_attrs(def_id);
        let link_name = attr::first_attr_value_str_by_name(&attrs, "link_name")
            .unwrap_or(name)
            .as_str();

        let args_res: EvalResult<Vec<Value>> = arg_operands.iter()
            .map(|arg| self.eval_operand(arg))
            .collect();
        let args = args_res?;

        let usize = self.tcx.types.usize;

        match &link_name[..] {
            "malloc" => {
                let size = self.value_to_primval(args[0], usize)?.to_u64()?;
                if size == 0 {
                    self.write_null(dest, dest_ty)?;
                } else {
                    let align = self.memory.pointer_size();
                    let ptr = self.memory.allocate(size, align, Kind::C)?;
                    self.write_primval(dest, PrimVal::Ptr(ptr), dest_ty)?;
                }
            }

            "free" => {
                let ptr = args[0].into_ptr(&mut self.memory)?;
                if !ptr.is_null()? {
                    self.memory.deallocate(ptr.to_ptr()?, None, Kind::C)?;
                }
            }

            "syscall" => {
                match self.value_to_primval(args[0], usize)?.to_u64()? {
                    511 => return Err(EvalError::Unimplemented("miri does not support random number generators".to_owned())),
                    id => return Err(EvalError::Unimplemented(format!("miri does not support syscall id {}", id))),
                }
            }

            "dlsym" => {
                let _handle = args[0].into_ptr(&mut self.memory)?;
                let symbol = args[1].into_ptr(&mut self.memory)?.to_ptr()?;
                let symbol_name = self.memory.read_c_str(symbol)?;
                let err = format!("bad c unicode symbol: {:?}", symbol_name);
                let symbol_name = ::std::str::from_utf8(symbol_name).unwrap_or(&err);
                return Err(EvalError::Unimplemented(format!("miri does not support dynamically loading libraries (requested symbol: {})", symbol_name)));
            }

            "__rust_maybe_catch_panic" => {
                // fn __rust_maybe_catch_panic(f: fn(*mut u8), data: *mut u8, data_ptr: *mut usize, vtable_ptr: *mut usize) -> u32
                // We abort on panic, so not much is going on here, but we still have to call the closure
                let u8_ptr_ty = self.tcx.mk_mut_ptr(self.tcx.types.u8);
                let f = args[0].into_ptr(&mut self.memory)?.to_ptr()?;
                let data = args[1].into_ptr(&mut self.memory)?;
                let f_instance = self.memory.get_fn(f)?;
                self.write_null(dest, dest_ty)?;

                // Now we make a function call.  TODO: Consider making this re-usable?  EvalContext::step does sth. similar for the TLS dtors,
                // and of course eval_main.
                let mir = self.load_mir(f_instance.def)?;
                self.push_stack_frame(
                    f_instance,
                    mir.span,
                    mir,
                    Lvalue::undef(),
                    StackPopCleanup::Goto(dest_block),
                )?;

                let arg_local = self.frame().mir.args_iter().next().ok_or(EvalError::AbiViolation("Argument to __rust_maybe_catch_panic does not take enough arguments.".to_owned()))?;
                let arg_dest = self.eval_lvalue(&mir::Lvalue::Local(arg_local))?;
                self.write_ptr(arg_dest, data, u8_ptr_ty)?;

                // We ourselves return 0
                self.write_null(dest, dest_ty)?;

                // Don't fall through
                return Ok(());
            }

            "__rust_start_panic" => {
                return Err(EvalError::Panic);
            }

            "memcmp" => {
                let left = args[0].into_ptr(&mut self.memory)?;
                let right = args[1].into_ptr(&mut self.memory)?;
                let n = self.value_to_primval(args[2], usize)?.to_u64()?;

                let result = {
                    let left_bytes = self.memory.read_bytes(left, n)?;
                    let right_bytes = self.memory.read_bytes(right, n)?;

                    use std::cmp::Ordering::*;
                    match left_bytes.cmp(right_bytes) {
                        Less => -1i8,
                        Equal => 0,
                        Greater => 1,
                    }
                };

                self.write_primval(dest, PrimVal::Bytes(result as u128), dest_ty)?;
            }

            "memrchr" => {
                let ptr = args[0].into_ptr(&mut self.memory)?;
                let val = self.value_to_primval(args[1], usize)?.to_u64()? as u8;
                let num = self.value_to_primval(args[2], usize)?.to_u64()?;
                if let Some(idx) = self.memory.read_bytes(ptr, num)?.iter().rev().position(|&c| c == val) {
                    let new_ptr = ptr.offset(num - idx as u64 - 1, &self)?;
                    self.write_ptr(dest, new_ptr, dest_ty)?;
                } else {
                    self.write_null(dest, dest_ty)?;
                }
            }

            "memchr" => {
                let ptr = args[0].into_ptr(&mut self.memory)?;
                let val = self.value_to_primval(args[1], usize)?.to_u64()? as u8;
                let num = self.value_to_primval(args[2], usize)?.to_u64()?;
                if let Some(idx) = self.memory.read_bytes(ptr, num)?.iter().position(|&c| c == val) {
                    let new_ptr = ptr.offset(idx as u64, &self)?;
                    self.write_ptr(dest, new_ptr, dest_ty)?;
                } else {
                    self.write_null(dest, dest_ty)?;
                }
            }

            "getenv" => {
                let result = {
                    let name_ptr = args[0].into_ptr(&mut self.memory)?.to_ptr()?;
                    let name = self.memory.read_c_str(name_ptr)?;
                    match self.machine_data.env_vars.get(name) {
                        Some(&var) => PrimVal::Ptr(var),
                        None => PrimVal::Bytes(0),
                    }
                };
                self.write_primval(dest, result, dest_ty)?;
            }

            "unsetenv" => {
                let mut success = None;
                {
                    let name_ptr = args[0].into_ptr(&mut self.memory)?;
                    if !name_ptr.is_null()? {
                        let name = self.memory.read_c_str(name_ptr.to_ptr()?)?;
                        if !name.is_empty() && !name.contains(&b'=') {
                            success = Some(self.machine_data.env_vars.remove(name));
                        }
                    }
                }
                if let Some(old) = success {
                    if let Some(var) = old {
                        self.memory.deallocate(var, None, Kind::Env)?;
                    }
                    self.write_null(dest, dest_ty)?;
                } else {
                    self.write_primval(dest, PrimVal::from_i128(-1), dest_ty)?;
                }
            }

            "setenv" => {
                let mut new = None;
                {
                    let name_ptr = args[0].into_ptr(&mut self.memory)?;
                    let value_ptr = args[1].into_ptr(&mut self.memory)?.to_ptr()?;
                    let value = self.memory.read_c_str(value_ptr)?;
                    if !name_ptr.is_null()? {
                        let name = self.memory.read_c_str(name_ptr.to_ptr()?)?;
                        if !name.is_empty() && !name.contains(&b'=') {
                            new = Some((name.to_owned(), value.to_owned()));
                        }
                    }
                }
                if let Some((name, value)) = new {
                    // +1 for the null terminator
                    let value_copy = self.memory.allocate((value.len() + 1) as u64, 1, Kind::Env)?;
                    self.memory.write_bytes(value_copy.into(), &value)?;
                    let trailing_zero_ptr = value_copy.offset(value.len() as u64, &self)?.into();
                    self.memory.write_bytes(trailing_zero_ptr, &[0])?;
                    if let Some(var) = self.machine_data.env_vars.insert(name.to_owned(), value_copy) {
                        self.memory.deallocate(var, None, Kind::Env)?;
                    }
                    self.write_null(dest, dest_ty)?;
                } else {
                    self.write_primval(dest, PrimVal::from_i128(-1), dest_ty)?;
                }
            }

            "write" => {
                let fd = self.value_to_primval(args[0], usize)?.to_u64()?;
                let buf = args[1].into_ptr(&mut self.memory)?;
                let n = self.value_to_primval(args[2], usize)?.to_u64()?;
                trace!("Called write({:?}, {:?}, {:?})", fd, buf, n);
                let result = if fd == 1 || fd == 2 { // stdout/stderr
                    use std::io::{self, Write};
                
                    let buf_cont = self.memory.read_bytes(buf, n)?;
                    let res = if fd == 1 { io::stdout().write(buf_cont) } else { io::stderr().write(buf_cont) };
                    match res { Ok(n) => n as isize, Err(_) => -1 }
                } else {
                    info!("Ignored output to FD {}", fd);
                    n as isize // pretend it all went well
                }; // now result is the value we return back to the program
                self.write_primval(dest, PrimVal::Bytes(result as u128), dest_ty)?;
            }

            "strlen" => {
                let ptr = args[0].into_ptr(&mut self.memory)?.to_ptr()?;
                let n = self.memory.read_c_str(ptr)?.len();
                self.write_primval(dest, PrimVal::Bytes(n as u128), dest_ty)?;
            }

            // Some things needed for sys::thread initialization to go through
            "signal" | "sigaction" | "sigaltstack" => {
                self.write_primval(dest, PrimVal::Bytes(0), dest_ty)?;
            }

            "sysconf" => {
                let name = self.value_to_primval(args[0], usize)?.to_u64()?;
                trace!("sysconf() called with name {}", name);
                // cache the sysconf integers via miri's global cache
                let paths = &[
                    (&["libc", "_SC_PAGESIZE"], PrimVal::Bytes(4096)),
                    (&["libc", "_SC_GETPW_R_SIZE_MAX"], PrimVal::from_i128(-1)),
                ];
                let mut result = None;
                for &(path, path_value) in paths {
                    if let Ok(instance) = self.resolve_path(path) {
                        let cid = GlobalId { instance, promoted: None };
                        // compute global if not cached
                        let val = match self.globals.get(&cid).map(|glob| glob.value) {
                            Some(value) => self.value_to_primval(value, usize)?.to_u64()?,
                            None => eval_body_as_primval(self.tcx, instance)?.0.to_u64()?,
                        };
                        if val == name {
                            result = Some(path_value);
                            break;
                        }
                    }
                }
                if let Some(result) = result {
                    self.write_primval(dest, result, dest_ty)?;
                } else {
                    return Err(EvalError::Unimplemented(format!("Unimplemented sysconf name: {}", name)));
                }
            }

            // Hook pthread calls that go to the thread-local storage memory subsystem
            "pthread_key_create" => {
                let key_ptr = args[0].into_ptr(&mut self.memory)?;

                // Extract the function type out of the signature (that seems easier than constructing it ourselves...)
                let dtor = match args[1].into_ptr(&mut self.memory)?.into_inner_primval() {
                    PrimVal::Ptr(dtor_ptr) => Some(self.memory.get_fn(dtor_ptr)?),
                    PrimVal::Bytes(0) => None,
                    PrimVal::Bytes(_) => return Err(EvalError::ReadBytesAsPointer),
                    PrimVal::Undef => return Err(EvalError::ReadUndefBytes),
                };

                // Figure out how large a pthread TLS key actually is. This is libc::pthread_key_t.
                let key_type = self.operand_ty(&arg_operands[0]).builtin_deref(true, ty::LvaluePreference::NoPreference)
                                   .ok_or(EvalError::AbiViolation("Wrong signature used for pthread_key_create: First argument must be a raw pointer.".to_owned()))?.ty;
                let key_size = {
                    let layout = self.type_layout(key_type)?;
                    layout.size(&self.tcx.data_layout)
                };

                // Create key and write it into the memory where key_ptr wants it
                let key = self.memory.create_tls_key(dtor) as u128;
                if key_size.bits() < 128 && key >= (1u128 << key_size.bits() as u128) {
                    return Err(EvalError::OutOfTls);
                }
                // TODO: Does this need checking for alignment?
                self.memory.write_uint(key_ptr.to_ptr()?, key, key_size.bytes())?;

                // Return success (0)
                self.write_null(dest, dest_ty)?;
            }
            "pthread_key_delete" => {
                // The conversion into TlsKey here is a little fishy, but should work as long as usize >= libc::pthread_key_t
                let key = self.value_to_primval(args[0], usize)?.to_u64()? as TlsKey;
                self.memory.delete_tls_key(key)?;
                // Return success (0)
                self.write_null(dest, dest_ty)?;
            }
            "pthread_getspecific" => {
                // The conversion into TlsKey here is a little fishy, but should work as long as usize >= libc::pthread_key_t
                let key = self.value_to_primval(args[0], usize)?.to_u64()? as TlsKey;
                let ptr = self.memory.load_tls(key)?;
                self.write_ptr(dest, ptr, dest_ty)?;
            }
            "pthread_setspecific" => {
                // The conversion into TlsKey here is a little fishy, but should work as long as usize >= libc::pthread_key_t
                let key = self.value_to_primval(args[0], usize)?.to_u64()? as TlsKey;
                let new_ptr = args[1].into_ptr(&mut self.memory)?;
                self.memory.store_tls(key, new_ptr)?;
                
                // Return success (0)
                self.write_null(dest, dest_ty)?;
            }

            // Stub out all the other pthread calls to just return 0
            link_name if link_name.starts_with("pthread_") => {
                warn!("ignoring C ABI call: {}", link_name);
                self.write_null(dest, dest_ty)?;
            },

            _ => {
                return Err(EvalError::Unimplemented(format!("can't call C ABI function: {}", link_name)));
            }
        }

        // Since we pushed no stack frame, the main loop will act
        // as if the call just completed and it's returning to the
        // current frame.
        self.dump_local(dest);
        self.goto_block(dest_block);
        Ok(())
    }

    fn finish(&mut self) -> EvalResult<'tcx> {
        let mut dtor = self.memory.fetch_tls_dtor(None)?;
        // FIXME: replace loop by some structure that works with stepping
        while let Some((instance, ptr, key)) = dtor {
            trace!("Running TLS dtor {:?} on {:?}", instance, ptr);
            // TODO: Potentially, this has to support all the other possible instances? See eval_fn_call in terminator/mod.rs
            let mir = self.load_mir(instance.def)?;
            self.push_stack_frame(
                instance,
                mir.span,
                mir,
                Lvalue::undef(),
                StackPopCleanup::None,
            )?;
            let arg_local = self.frame().mir.args_iter().next().ok_or(EvalError::AbiViolation("TLS dtor does not take enough arguments.".to_owned()))?;
            let dest = self.eval_lvalue(&mir::Lvalue::Local(arg_local))?;
            let ty = self.tcx.mk_mut_ptr(self.tcx.types.u8);
            self.write_ptr(dest, ptr, ty)?;

            // step until out of stackframes
            while self.step()? {}

            dtor = match self.memory.fetch_tls_dtor(Some(key))? {
                dtor @ Some(_) => dtor,
                None => self.memory.fetch_tls_dtor(None)?,
            };
        }
        Ok(())
    }

    /// Get an instance for a path.
    fn resolve_path(&self, path: &[&str]) -> EvalResult<'tcx, ty::Instance<'tcx>> {
        let cstore = &self.tcx.sess.cstore;

        let crates = cstore.crates();
        crates.iter()
            .find(|&&krate| cstore.crate_name(krate) == path[0])
            .and_then(|krate| {
                let krate = DefId {
                    krate: *krate,
                    index: CRATE_DEF_INDEX,
                };
                let mut items = cstore.item_children(krate, self.tcx.sess);
                let mut path_it = path.iter().skip(1).peekable();

                while let Some(segment) = path_it.next() {
                    for item in &mem::replace(&mut items, vec![]) {
                        if item.ident.name == *segment {
                            if path_it.peek().is_none() {
                                return Some(ty::Instance::mono(self.tcx, item.def.def_id()));
                            }

                            items = cstore.item_children(item.def.def_id(), self.tcx.sess);
                            break;
                        }
                    }
                }
                None
            })
            .ok_or_else(|| {
                let path = path.iter()
                    .map(|&s| s.to_owned())
                    .collect();
                EvalError::PathNotFound(path)
            })
    }

    fn call_missing_fn(
        &mut self,
        instance: ty::Instance<'tcx>,
        destination: Option<(Lvalue<'tcx>, mir::BasicBlock)>,
        arg_operands: &[mir::Operand<'tcx>],
        sig: ty::FnSig<'tcx>,
        path: String,
    ) -> EvalResult<'tcx> {
        // In some cases in non-MIR libstd-mode, not having a destination is legit.  Handle these early.
        match &path[..] {
            "std::panicking::rust_panic_with_hook" |
            "std::rt::begin_panic_fmt" => return Err(EvalError::Panic),
            _ => {},
        }

        let dest_ty = sig.output();
        let (dest, dest_block) = destination.ok_or_else(|| EvalError::NoMirFor(path.clone()))?;

        if sig.abi == Abi::C {
            // An external C function
            // TODO: That functions actually has a similar preamble to what follows here.  May make sense to
            // unify these two mechanisms for "hooking into missing functions".
            self.call_c_abi(instance.def_id(), arg_operands, dest, dest_ty, dest_block)?;
            return Ok(());
        }

        let args_res: EvalResult<Vec<Value>> = arg_operands.iter()
            .map(|arg| self.eval_operand(arg))
            .collect();
        let args = args_res?;

        let usize = self.tcx.types.usize;
    
        match &path[..] {
            // Allocators are magic.  They have no MIR, even when the rest of libstd does.
            "alloc::heap::::__rust_alloc" => {
                let size = self.value_to_primval(args[0], usize)?.to_u64()?;
                let align = self.value_to_primval(args[1], usize)?.to_u64()?;
                if size == 0 {
                    return Err(EvalError::HeapAllocZeroBytes);
                }
                if !align.is_power_of_two() {
                    return Err(EvalError::HeapAllocNonPowerOfTwoAlignment(align));
                }
                let ptr = self.memory.allocate(size, align, Kind::Rust)?;
                self.write_primval(dest, PrimVal::Ptr(ptr), dest_ty)?;
            }
            "alloc::heap::::__rust_alloc_zeroed" => {
                let size = self.value_to_primval(args[0], usize)?.to_u64()?;
                let align = self.value_to_primval(args[1], usize)?.to_u64()?;
                if size == 0 {
                    return Err(EvalError::HeapAllocZeroBytes);
                }
                if !align.is_power_of_two() {
                    return Err(EvalError::HeapAllocNonPowerOfTwoAlignment(align));
                }
                let ptr = self.memory.allocate(size, align, Kind::Rust)?;
                self.memory.write_repeat(ptr.into(), 0, size)?;
                self.write_primval(dest, PrimVal::Ptr(ptr), dest_ty)?;
            }
            "alloc::heap::::__rust_dealloc" => {
                let ptr = args[0].into_ptr(&mut self.memory)?.to_ptr()?;
                let old_size = self.value_to_primval(args[1], usize)?.to_u64()?;
                let align = self.value_to_primval(args[2], usize)?.to_u64()?;
                if old_size == 0 {
                    return Err(EvalError::HeapAllocZeroBytes);
                }
                if !align.is_power_of_two() {
                    return Err(EvalError::HeapAllocNonPowerOfTwoAlignment(align));
                }
                self.memory.deallocate(ptr, Some((old_size, align)), Kind::Rust)?;
            }
            "alloc::heap::::__rust_realloc" => {
                let ptr = args[0].into_ptr(&mut self.memory)?.to_ptr()?;
                let old_size = self.value_to_primval(args[1], usize)?.to_u64()?;
                let old_align = self.value_to_primval(args[2], usize)?.to_u64()?;
                let new_size = self.value_to_primval(args[3], usize)?.to_u64()?;
                let new_align = self.value_to_primval(args[4], usize)?.to_u64()?;
                if old_size == 0 || new_size == 0 {
                    return Err(EvalError::HeapAllocZeroBytes);
                }
                if !old_align.is_power_of_two() {
                    return Err(EvalError::HeapAllocNonPowerOfTwoAlignment(old_align));
                }
                if !new_align.is_power_of_two() {
                    return Err(EvalError::HeapAllocNonPowerOfTwoAlignment(new_align));
                }
                let new_ptr = self.memory.reallocate(ptr, old_size, old_align, new_size, new_align, Kind::Rust)?;
                self.write_primval(dest, PrimVal::Ptr(new_ptr), dest_ty)?;
            }

            // A Rust function is missing, which means we are running with MIR missing for libstd (or other dependencies).
            // Still, we can make many things mostly work by "emulating" or ignoring some functions.
            "std::io::_print" => {
                trace!("Ignoring output.  To run programs that print, make sure you have a libstd with full MIR.");
            }
            "std::thread::Builder::new" => return Err(EvalError::Unimplemented("miri does not support threading".to_owned())),
            "std::env::args" => return Err(EvalError::Unimplemented("miri does not support program arguments".to_owned())),
            "std::panicking::panicking" |
            "std::rt::panicking" => {
                // we abort on panic -> `std::rt::panicking` always returns false
                let bool = self.tcx.types.bool;
                self.write_primval(dest, PrimVal::from_bool(false), bool)?;
            }
            _ => return Err(EvalError::NoMirFor(path)),
        }

        // Since we pushed no stack frame, the main loop will act
        // as if the call just completed and it's returning to the
        // current frame.
        self.dump_local(dest);
        self.goto_block(dest_block);
        return Ok(());
    }
}

trait MemoryExt<'tcx> {
    fn create_tls_key(&mut self, dtor: Option<ty::Instance<'tcx>>) -> TlsKey;
    fn delete_tls_key(&mut self, key: TlsKey) -> EvalResult<'tcx>;
    fn load_tls(&mut self, key: TlsKey) -> EvalResult<'tcx, Pointer>;
    fn store_tls(&mut self, key: TlsKey, new_data: Pointer) -> EvalResult<'tcx>;
    fn fetch_tls_dtor(&mut self, key: Option<TlsKey>) -> EvalResult<'tcx, Option<(ty::Instance<'tcx>, Pointer, TlsKey)>>;
}

impl<'a, 'tcx: 'a> MemoryExt<'tcx> for Memory<'a, 'tcx, Evaluator> {
    fn create_tls_key(&mut self, dtor: Option<ty::Instance<'tcx>>) -> TlsKey {
        let new_key = self.data.next_thread_local;
        self.data.next_thread_local += 1;
        self.data.thread_local.insert(new_key, TlsEntry { data: Pointer::null(), dtor });
        trace!("New TLS key allocated: {} with dtor {:?}", new_key, dtor);
        return new_key;
    }

    fn delete_tls_key(&mut self, key: TlsKey) -> EvalResult<'tcx> {
        return match self.data.thread_local.remove(&key) {
            Some(_) => {
                trace!("TLS key {} removed", key);
                Ok(())
            },
            None => Err(EvalError::TlsOutOfBounds)
        }
    }

    fn load_tls(&mut self, key: TlsKey) -> EvalResult<'tcx, Pointer> {
        return match self.data.thread_local.get(&key) {
            Some(&TlsEntry { data, .. }) => {
                trace!("TLS key {} loaded: {:?}", key, data);
                Ok(data)
            },
            None => Err(EvalError::TlsOutOfBounds)
        }
    }

    fn store_tls(&mut self, key: TlsKey, new_data: Pointer) -> EvalResult<'tcx> {
        return match self.data.thread_local.get_mut(&key) {
            Some(&mut TlsEntry { ref mut data, .. }) => {
                trace!("TLS key {} stored: {:?}", key, new_data);
                *data = new_data;
                Ok(())
            },
            None => Err(EvalError::TlsOutOfBounds)
        }
    }
    
    /// Returns a dtor, its argument and its index, if one is supposed to run
    ///
    /// An optional destructor function may be associated with each key value.
    /// At thread exit, if a key value has a non-NULL destructor pointer,
    /// and the thread has a non-NULL value associated with that key,
    /// the value of the key is set to NULL, and then the function pointed
    /// to is called with the previously associated value as its sole argument.
    /// The order of destructor calls is unspecified if more than one destructor
    /// exists for a thread when it exits.
    ///
    /// If, after all the destructors have been called for all non-NULL values
    /// with associated destructors, there are still some non-NULL values with
    /// associated destructors, then the process is repeated.
    /// If, after at least {PTHREAD_DESTRUCTOR_ITERATIONS} iterations of destructor
    /// calls for outstanding non-NULL values, there are still some non-NULL values
    /// with associated destructors, implementations may stop calling destructors,
    /// or they may continue calling destructors until no non-NULL values with
    /// associated destructors exist, even though this might result in an infinite loop.
    fn fetch_tls_dtor(&mut self, key: Option<TlsKey>) -> EvalResult<'tcx, Option<(ty::Instance<'tcx>, Pointer, TlsKey)>> {
        use std::collections::Bound::*;
        let start = match key {
            Some(key) => Excluded(key),
            None => Unbounded,
        };
        for (&key, &mut TlsEntry { ref mut data, dtor }) in self.data.thread_local.range_mut((start, Unbounded)) {
            if !data.is_null()? {
                if let Some(dtor) = dtor {
                    let ret = Some((dtor, *data, key));
                    *data = Pointer::null();
                    return Ok(ret);
                }
            }
        }
        return Ok(None);
    }
}

impl<'tcx> Machine<'tcx> for Evaluator {
    type Data = EvaluatorData;
    type MemoryData = MemoryData<'tcx>;
    /// Returns Ok() when the function was handled, fail otherwise
    fn call_missing_fn<'a>(
        ecx: &mut EvalContext<'a, 'tcx, Self>,
        instance: ty::Instance<'tcx>,
        destination: Option<(Lvalue<'tcx>, mir::BasicBlock)>,
        arg_operands: &[mir::Operand<'tcx>],
        sig: ty::FnSig<'tcx>,
        path: String,
    ) -> EvalResult<'tcx> {
        ecx.call_missing_fn(instance, destination, arg_operands, sig, path)
    }
}
