use rustc::ty;
use rustc::ty::layout::{Align, LayoutOf, Size};
use rustc::hir::def_id::{DefId, CRATE_DEF_INDEX};
use rustc::mir;
use syntax::attr;

use std::mem;

use super::*;

use tls::MemoryExt;

use super::memory::MemoryKind;

pub trait EvalContextExt<'tcx, 'mir> {
    /// Emulate calling a foreign item, fail if the item is not supported.
    /// This function will handle `goto_block` if needed.
    fn emulate_foreign_item(
        &mut self,
        def_id: DefId,
        args: &[OpTy<'tcx>],
        dest: PlaceTy<'tcx>,
        ret: mir::BasicBlock,
    ) -> EvalResult<'tcx>;

    fn resolve_path(&self, path: &[&str]) -> EvalResult<'tcx, ty::Instance<'tcx>>;

    /// Emulate a function that should have MIR but does not.
    /// This is solely to support execution without full MIR.
    /// Fail if emulating this function is not supported.
    /// This function will handle `goto_block` if needed.
    fn emulate_missing_fn(
        &mut self,
        path: String,
        args: &[OpTy<'tcx>],
        dest: Option<PlaceTy<'tcx>>,
        ret: Option<mir::BasicBlock>,
    ) -> EvalResult<'tcx>;

    fn find_fn(
        &mut self,
        instance: ty::Instance<'tcx>,
        args: &[OpTy<'tcx>],
        dest: Option<PlaceTy<'tcx>>,
        ret: Option<mir::BasicBlock>,
    ) -> EvalResult<'tcx, Option<&'mir mir::Mir<'tcx>>>;

    fn write_null(&mut self, dest: PlaceTy<'tcx>) -> EvalResult<'tcx>;
}

impl<'a, 'mir, 'tcx: 'mir + 'a> EvalContextExt<'tcx, 'mir> for EvalContext<'a, 'mir, 'tcx, super::Evaluator<'tcx>> {
    fn find_fn(
        &mut self,
        instance: ty::Instance<'tcx>,
        args: &[OpTy<'tcx>],
        dest: Option<PlaceTy<'tcx>>,
        ret: Option<mir::BasicBlock>,
    ) -> EvalResult<'tcx, Option<&'mir mir::Mir<'tcx>>> {
        trace!("eval_fn_call: {:#?}, {:?}", instance, dest.map(|place| *place));

        // first run the common hooks also supported by CTFE
        if self.hook_fn(instance, args, dest)? {
            self.goto_block(ret)?;
            return Ok(None);
        }
        // there are some more lang items we want to hook that CTFE does not hook (yet)
        if self.tcx.lang_items().align_offset_fn() == Some(instance.def.def_id()) {
            // FIXME: return a real value in case the target allocation has an
            // alignment bigger than the one requested
            let n = u128::max_value();
            let dest = dest.unwrap();
            let n = self.truncate(n, dest.layout);
            self.write_scalar(Scalar::from_uint(n, dest.layout.size), dest)?;
            self.goto_block(ret)?;
            return Ok(None);
        }

        // Try to see if we can do something about foreign items
        if self.tcx.is_foreign_item(instance.def_id()) {
            // An external function that we cannot find MIR for, but we can still run enough
            // of them to make miri viable.
            self.emulate_foreign_item(
                instance.def_id(),
                args,
                dest.unwrap(),
                ret.unwrap(),
            )?;
            // `goto_block` already handled
            return Ok(None);
        }

        // Otherwise we really want to see the MIR -- but if we do not have it, maybe we can
        // emulate something. This is a HACK to support running without a full-MIR libstd.
        let mir = match self.load_mir(instance.def) {
            Ok(mir) => mir,
            Err(EvalError { kind: EvalErrorKind::NoMirFor(path), .. }) => {
                self.emulate_missing_fn(
                    path,
                    args,
                    dest,
                    ret,
                )?;
                // `goto_block` already handled
                return Ok(None);
            }
            Err(other) => return Err(other),
        };

        Ok(Some(mir))
    }

    fn emulate_foreign_item(
        &mut self,
        def_id: DefId,
        args: &[OpTy<'tcx>],
        dest: PlaceTy<'tcx>,
        ret: mir::BasicBlock,
    ) -> EvalResult<'tcx> {
        let attrs = self.tcx.get_attrs(def_id);
        let link_name = match attr::first_attr_value_str_by_name(&attrs, "link_name") {
            Some(name) => name.as_str(),
            None => self.tcx.item_name(def_id).as_str(),
        };

        match &link_name[..] {
            "malloc" => {
                let size = self.read_scalar(args[0])?.to_usize(&self)?;
                if size == 0 {
                    self.write_null(dest)?;
                } else {
                    let align = self.tcx.data_layout.pointer_align;
                    let ptr = self.memory.allocate(Size::from_bytes(size), align, MemoryKind::C.into())?;
                    self.write_scalar(Scalar::Ptr(ptr), dest)?;
                }
            }

            "free" => {
                let ptr = self.read_scalar(args[0])?.not_undef()?;
                if !ptr.is_null() {
                    self.memory.deallocate(
                        ptr.to_ptr()?,
                        None,
                        MemoryKind::C.into(),
                    )?;
                }
            }

            "__rust_alloc" => {
                let size = self.read_scalar(args[0])?.to_usize(&self)?;
                let align = self.read_scalar(args[1])?.to_usize(&self)?;
                if size == 0 {
                    return err!(HeapAllocZeroBytes);
                }
                if !align.is_power_of_two() {
                    return err!(HeapAllocNonPowerOfTwoAlignment(align));
                }
                let ptr = self.memory.allocate(Size::from_bytes(size),
                                               Align::from_bytes(align, align).unwrap(),
                                               MemoryKind::Rust.into())?;
                self.write_scalar(Scalar::Ptr(ptr), dest)?;
            }
            "__rust_alloc_zeroed" => {
                let size = self.read_scalar(args[0])?.to_usize(&self)?;
                let align = self.read_scalar(args[1])?.to_usize(&self)?;
                if size == 0 {
                    return err!(HeapAllocZeroBytes);
                }
                if !align.is_power_of_two() {
                    return err!(HeapAllocNonPowerOfTwoAlignment(align));
                }
                let ptr = self.memory.allocate(Size::from_bytes(size),
                                               Align::from_bytes(align, align).unwrap(),
                                               MemoryKind::Rust.into())?;
                self.memory.write_repeat(ptr.into(), 0, Size::from_bytes(size))?;
                self.write_scalar(Scalar::Ptr(ptr), dest)?;
            }
            "__rust_dealloc" => {
                let ptr = self.read_scalar(args[0])?.to_ptr()?;
                let old_size = self.read_scalar(args[1])?.to_usize(&self)?;
                let align = self.read_scalar(args[2])?.to_usize(&self)?;
                if old_size == 0 {
                    return err!(HeapAllocZeroBytes);
                }
                if !align.is_power_of_two() {
                    return err!(HeapAllocNonPowerOfTwoAlignment(align));
                }
                self.memory.deallocate(
                    ptr,
                    Some((Size::from_bytes(old_size), Align::from_bytes(align, align).unwrap())),
                    MemoryKind::Rust.into(),
                )?;
            }
            "__rust_realloc" => {
                let ptr = self.read_scalar(args[0])?.to_ptr()?;
                let old_size = self.read_scalar(args[1])?.to_usize(&self)?;
                let align = self.read_scalar(args[2])?.to_usize(&self)?;
                let new_size = self.read_scalar(args[3])?.to_usize(&self)?;
                if old_size == 0 || new_size == 0 {
                    return err!(HeapAllocZeroBytes);
                }
                if !align.is_power_of_two() {
                    return err!(HeapAllocNonPowerOfTwoAlignment(align));
                }
                let new_ptr = self.memory.reallocate(
                    ptr,
                    Size::from_bytes(old_size),
                    Align::from_bytes(align, align).unwrap(),
                    Size::from_bytes(new_size),
                    Align::from_bytes(align, align).unwrap(),
                    MemoryKind::Rust.into(),
                )?;
                self.write_scalar(Scalar::Ptr(new_ptr), dest)?;
            }

            "syscall" => {
                // TODO: read `syscall` ids like `sysconf` ids and
                // figure out some way to actually process some of them
                //
                // libc::syscall(NR_GETRANDOM, buf.as_mut_ptr(), buf.len(), GRND_NONBLOCK)
                // is called if a `HashMap` is created the regular way.
                match self.read_scalar(args[0])?.to_usize(&self)? {
                    318 | 511 => {
                        return err!(Unimplemented(
                            "miri does not support random number generators".to_owned(),
                        ))
                    }
                    id => {
                        return err!(Unimplemented(
                            format!("miri does not support syscall id {}", id),
                        ))
                    }
                }
            }

            "dlsym" => {
                let _handle = self.read_scalar(args[0])?;
                let symbol = self.read_scalar(args[1])?.to_ptr()?;
                let symbol_name = self.memory.read_c_str(symbol)?;
                let err = format!("bad c unicode symbol: {:?}", symbol_name);
                let symbol_name = ::std::str::from_utf8(symbol_name).unwrap_or(&err);
                return err!(Unimplemented(format!(
                    "miri does not support dynamically loading libraries (requested symbol: {})",
                    symbol_name
                )));
            }

            "__rust_maybe_catch_panic" => {
                // fn __rust_maybe_catch_panic(f: fn(*mut u8), data: *mut u8, data_ptr: *mut usize, vtable_ptr: *mut usize) -> u32
                // We abort on panic, so not much is going on here, but we still have to call the closure
                let f = self.read_scalar(args[0])?.to_ptr()?;
                let data = self.read_scalar(args[1])?.not_undef()?;
                let f_instance = self.memory.get_fn(f)?;
                self.write_null(dest)?;
                trace!("__rust_maybe_catch_panic: {:?}", f_instance);

                // Now we make a function call.  TODO: Consider making this re-usable?  EvalContext::step does sth. similar for the TLS dtors,
                // and of course eval_main.
                let mir = self.load_mir(f_instance.def)?;
                let closure_dest = Place::null(&self);
                self.push_stack_frame(
                    f_instance,
                    mir.span,
                    mir,
                    closure_dest,
                    StackPopCleanup::Goto(Some(ret)), // directly return to caller
                )?;
                let mut args = self.frame().mir.args_iter();

                let arg_local = args.next().ok_or_else(||
                    EvalErrorKind::AbiViolation(
                        "Argument to __rust_maybe_catch_panic does not take enough arguments."
                            .to_owned(),
                    ),
                )?;
                let arg_dest = self.eval_place(&mir::Place::Local(arg_local))?;
                self.write_scalar(data, arg_dest)?;

                assert!(args.next().is_none(), "__rust_maybe_catch_panic argument has more arguments than expected");

                // We ourselves will return 0, eventually (because we will not return if we paniced)
                self.write_null(dest)?;

                // Don't fall through, we do NOT want to `goto_block`!
                return Ok(());
            }

            "__rust_start_panic" =>
                return err!(MachineError("the evaluated program panicked".to_string())),

            "memcmp" => {
                let left = self.read_scalar(args[0])?.not_undef()?;
                let right = self.read_scalar(args[1])?.not_undef()?;
                let n = Size::from_bytes(self.read_scalar(args[2])?.to_usize(&self)?);

                let result = {
                    let left_bytes = self.memory.read_bytes(left, n)?;
                    let right_bytes = self.memory.read_bytes(right, n)?;

                    use std::cmp::Ordering::*;
                    match left_bytes.cmp(right_bytes) {
                        Less => -1i32,
                        Equal => 0,
                        Greater => 1,
                    }
                };

                self.write_scalar(
                    Scalar::from_i32(result),
                    dest,
                )?;
            }

            "memrchr" => {
                let ptr = self.read_scalar(args[0])?.not_undef()?;
                let val = self.read_scalar(args[1])?.to_bytes()? as u8;
                let num = self.read_scalar(args[2])?.to_usize(&self)?;
                if let Some(idx) = self.memory.read_bytes(ptr, Size::from_bytes(num))?.iter().rev().position(
                    |&c| c == val,
                )
                {
                    let new_ptr = ptr.ptr_offset(Size::from_bytes(num - idx as u64 - 1), &self)?;
                    self.write_scalar(new_ptr, dest)?;
                } else {
                    self.write_null(dest)?;
                }
            }

            "memchr" => {
                let ptr = self.read_scalar(args[0])?.not_undef()?;
                let val = self.read_scalar(args[1])?.to_bytes()? as u8;
                let num = self.read_scalar(args[2])?.to_usize(&self)?;
                if let Some(idx) = self.memory.read_bytes(ptr, Size::from_bytes(num))?.iter().position(
                    |&c| c == val,
                )
                {
                    let new_ptr = ptr.ptr_offset(Size::from_bytes(idx as u64), &self)?;
                    self.write_scalar(new_ptr, dest)?;
                } else {
                    self.write_null(dest)?;
                }
            }

            "getenv" => {
                let result = {
                    let name_ptr = self.read_scalar(args[0])?.to_ptr()?;
                    let name = self.memory.read_c_str(name_ptr)?;
                    match self.machine.env_vars.get(name) {
                        Some(&var) => Scalar::Ptr(var),
                        None => Scalar::null(self.memory.pointer_size()),
                    }
                };
                self.write_scalar(result, dest)?;
            }

            "unsetenv" => {
                let mut success = None;
                {
                    let name_ptr = self.read_scalar(args[0])?.not_undef()?;
                    if !name_ptr.is_null() {
                        let name = self.memory.read_c_str(name_ptr.to_ptr()?)?;
                        if !name.is_empty() && !name.contains(&b'=') {
                            success = Some(self.machine.env_vars.remove(name));
                        }
                    }
                }
                if let Some(old) = success {
                    if let Some(var) = old {
                        self.memory.deallocate(var, None, MemoryKind::Env.into())?;
                    }
                    self.write_null(dest)?;
                } else {
                    self.write_scalar(Scalar::from_int(-1, dest.layout.size), dest)?;
                }
            }

            "setenv" => {
                let mut new = None;
                {
                    let name_ptr = self.read_scalar(args[0])?.not_undef()?;
                    let value_ptr = self.read_scalar(args[1])?.to_ptr()?;
                    let value = self.memory.read_c_str(value_ptr)?;
                    if !name_ptr.is_null() {
                        let name = self.memory.read_c_str(name_ptr.to_ptr()?)?;
                        if !name.is_empty() && !name.contains(&b'=') {
                            new = Some((name.to_owned(), value.to_owned()));
                        }
                    }
                }
                if let Some((name, value)) = new {
                    // +1 for the null terminator
                    let value_copy = self.memory.allocate(
                        Size::from_bytes((value.len() + 1) as u64),
                        Align::from_bytes(1, 1).unwrap(),
                        MemoryKind::Env.into(),
                    )?;
                    self.memory.write_bytes(value_copy.into(), &value)?;
                    let trailing_zero_ptr = value_copy.offset(Size::from_bytes(value.len() as u64), &self)?.into();
                    self.memory.write_bytes(trailing_zero_ptr, &[0])?;
                    if let Some(var) = self.machine.env_vars.insert(
                        name.to_owned(),
                        value_copy,
                    )
                    {
                        self.memory.deallocate(var, None, MemoryKind::Env.into())?;
                    }
                    self.write_null(dest)?;
                } else {
                    self.write_scalar(Scalar::from_int(-1, dest.layout.size), dest)?;
                }
            }

            "write" => {
                let fd = self.read_scalar(args[0])?.to_bytes()?;
                let buf = self.read_scalar(args[1])?.not_undef()?;
                let n = self.read_scalar(args[2])?.to_bytes()? as u64;
                trace!("Called write({:?}, {:?}, {:?})", fd, buf, n);
                let result = if fd == 1 || fd == 2 {
                    // stdout/stderr
                    use std::io::{self, Write};

                    let buf_cont = self.memory.read_bytes(buf, Size::from_bytes(n))?;
                    let res = if fd == 1 {
                        io::stdout().write(buf_cont)
                    } else {
                        io::stderr().write(buf_cont)
                    };
                    match res {
                        Ok(n) => n as i64,
                        Err(_) => -1,
                    }
                } else {
                    warn!("Ignored output to FD {}", fd);
                    n as i64 // pretend it all went well
                }; // now result is the value we return back to the program
                self.write_scalar(
                    Scalar::from_int(result, dest.layout.size),
                    dest,
                )?;
            }

            "strlen" => {
                let ptr = self.read_scalar(args[0])?.to_ptr()?;
                let n = self.memory.read_c_str(ptr)?.len();
                self.write_scalar(Scalar::from_uint(n as u64, dest.layout.size), dest)?;
            }

            // Some things needed for sys::thread initialization to go through
            "signal" | "sigaction" | "sigaltstack" => {
                self.write_scalar(Scalar::null(dest.layout.size), dest)?;
            }

            "sysconf" => {
                let name = self.read_scalar(args[0])?.to_i32()?;

                trace!("sysconf() called with name {}", name);
                // cache the sysconf integers via miri's global cache
                let paths = &[
                    (&["libc", "_SC_PAGESIZE"], Scalar::from_int(4096, dest.layout.size)),
                    (&["libc", "_SC_GETPW_R_SIZE_MAX"], Scalar::from_int(-1, dest.layout.size)),
                ];
                let mut result = None;
                for &(path, path_value) in paths {
                    if let Ok(instance) = self.resolve_path(path) {
                        let cid = GlobalId {
                            instance,
                            promoted: None,
                        };
                        let const_val = self.const_eval(cid)?;
                        let value = const_val.unwrap_bits(
                            self.tcx.tcx,
                            ty::ParamEnv::empty().and(self.tcx.types.i32)) as i32;
                        if value == name {
                            result = Some(path_value);
                            break;
                        }
                    }
                }
                if let Some(result) = result {
                    self.write_scalar(result, dest)?;
                } else {
                    return err!(Unimplemented(
                        format!("Unimplemented sysconf name: {}", name),
                    ));
                }
            }

            // Hook pthread calls that go to the thread-local storage memory subsystem
            "pthread_key_create" => {
                let key_ptr = self.read_scalar(args[0])?.to_ptr()?;

                // Extract the function type out of the signature (that seems easier than constructing it ourselves...)
                let dtor = match self.read_scalar(args[1])?.not_undef()? {
                    Scalar::Ptr(dtor_ptr) => Some(self.memory.get_fn(dtor_ptr)?),
                    Scalar::Bits { bits: 0, size } => {
                        assert_eq!(size as u64, self.memory.pointer_size().bytes());
                        None
                    },
                    Scalar::Bits { .. } => return err!(ReadBytesAsPointer),
                };

                // Figure out how large a pthread TLS key actually is. This is libc::pthread_key_t.
                let key_type = args[0].layout.ty.builtin_deref(true)
                                   .ok_or_else(|| EvalErrorKind::AbiViolation("Wrong signature used for pthread_key_create: First argument must be a raw pointer.".to_owned()))?.ty;
                let key_layout = self.layout_of(key_type)?;

                // Create key and write it into the memory where key_ptr wants it
                let key = self.memory.create_tls_key(dtor) as u128;
                if key_layout.size.bits() < 128 && key >= (1u128 << key_layout.size.bits() as u128) {
                    return err!(OutOfTls);
                }
                self.memory.write_scalar(
                    key_ptr,
                    key_layout.align,
                    Scalar::from_uint(key, key_layout.size).into(),
                    key_layout.size,
                )?;

                // Return success (0)
                self.write_null(dest)?;
            }
            "pthread_key_delete" => {
                let key = self.read_scalar(args[0])?.to_bytes()?;
                self.memory.delete_tls_key(key)?;
                // Return success (0)
                self.write_null(dest)?;
            }
            "pthread_getspecific" => {
                let key = self.read_scalar(args[0])?.to_bytes()?;
                let ptr = self.memory.load_tls(key)?;
                self.write_scalar(ptr, dest)?;
            }
            "pthread_setspecific" => {
                let key = self.read_scalar(args[0])?.to_bytes()?;
                let new_ptr = self.read_scalar(args[1])?.not_undef()?;
                self.memory.store_tls(key, new_ptr)?;

                // Return success (0)
                self.write_null(dest)?;
            }

            "_tlv_atexit" => {
                return err!(Unimplemented("Thread-local store is not fully supported on macOS".to_owned()));
            },

            // Determining stack base address
            "pthread_attr_init" | "pthread_attr_destroy" | "pthread_attr_get_np" |
            "pthread_getattr_np" | "pthread_self" => {
                self.write_null(dest)?;
            }
            "pthread_attr_getstack" => {
                // second argument is where we are supposed to write the stack size
                let ptr = self.ref_to_mplace(self.read_value(args[1])?)?;
                self.write_scalar(Scalar::from_int(0x80000, args[1].layout.size), ptr.into())?;
                // return 0
                self.write_null(dest)?;
            }

            // Stub out calls for condvar, mutex and rwlock to just return 0
            "pthread_mutexattr_init" | "pthread_mutexattr_settype" | "pthread_mutex_init" |
            "pthread_mutexattr_destroy" | "pthread_mutex_lock" | "pthread_mutex_unlock" |
            "pthread_mutex_destroy" | "pthread_rwlock_rdlock" | "pthread_rwlock_unlock" |
            "pthread_rwlock_wrlock" | "pthread_rwlock_destroy" | "pthread_condattr_init" |
            "pthread_condattr_setclock" | "pthread_cond_init" | "pthread_condattr_destroy" |
            "pthread_cond_destroy" => {
                self.write_null(dest)?;
            }

            "mmap" => {
                // This is a horrible hack, but well... the guard page mechanism calls mmap and expects a particular return value, so we give it that value
                let addr = self.read_scalar(args[0])?.not_undef()?;
                self.write_scalar(addr, dest)?;
            }

            // Windows API subs
            "AddVectoredExceptionHandler" => {
                // any non zero value works for the stdlib. This is just used for stackoverflows anyway
                self.write_scalar(Scalar::from_int(1, dest.layout.size), dest)?;
            },
            "InitializeCriticalSection" |
            "EnterCriticalSection" |
            "LeaveCriticalSection" |
            "DeleteCriticalSection" |
            "SetLastError" => {
                // Function does not return anything, nothing to do
            },
            "GetModuleHandleW" |
            "GetProcAddress" |
            "TryEnterCriticalSection" => {
                // pretend these do not exist/nothing happened, by returning zero
                self.write_null(dest)?;
            },
            "GetLastError" => {
                // this is c::ERROR_CALL_NOT_IMPLEMENTED
                self.write_scalar(Scalar::from_int(120, dest.layout.size), dest)?;
            },

            // Windows TLS
            "TlsAlloc" => {
                // This just creates a key; Windows does not natively support TLS dtors.

                // Create key and return it
                let key = self.memory.create_tls_key(None) as u128;

                // Figure out how large a TLS key actually is. This is c::DWORD.
                if dest.layout.size.bits() < 128 && key >= (1u128 << dest.layout.size.bits() as u128) {
                    return err!(OutOfTls);
                }
                self.write_scalar(Scalar::from_uint(key, dest.layout.size), dest)?;
            }
            "TlsGetValue" => {
                let key = self.read_scalar(args[0])?.to_bytes()?;
                let ptr = self.memory.load_tls(key)?;
                self.write_scalar(ptr, dest)?;
            }
            "TlsSetValue" => {
                let key = self.read_scalar(args[0])?.to_bytes()?;
                let new_ptr = self.read_scalar(args[1])?.not_undef()?;
                self.memory.store_tls(key, new_ptr)?;

                // Return success (1)
                self.write_scalar(Scalar::from_int(1, dest.layout.size), dest)?;
            }

            // We can't execute anything else
            _ => {
                return err!(Unimplemented(
                    format!("can't call foreign function: {}", link_name),
                ));
            }
        }

        self.goto_block(Some(ret))?;
        self.dump_place(*dest);
        Ok(())
    }

    /// Get an instance for a path.
    fn resolve_path(&self, path: &[&str]) -> EvalResult<'tcx, ty::Instance<'tcx>> {
        self.tcx
            .crates()
            .iter()
            .find(|&&krate| self.tcx.original_crate_name(krate) == path[0])
            .and_then(|krate| {
                let krate = DefId {
                    krate: *krate,
                    index: CRATE_DEF_INDEX,
                };
                let mut items = self.tcx.item_children(krate);
                let mut path_it = path.iter().skip(1).peekable();

                while let Some(segment) = path_it.next() {
                    for item in mem::replace(&mut items, Default::default()).iter() {
                        if item.ident.name == *segment {
                            if path_it.peek().is_none() {
                                return Some(ty::Instance::mono(self.tcx.tcx, item.def.def_id()));
                            }

                            items = self.tcx.item_children(item.def.def_id());
                            break;
                        }
                    }
                }
                None
            })
            .ok_or_else(|| {
                let path = path.iter().map(|&s| s.to_owned()).collect();
                EvalErrorKind::PathNotFound(path).into()
            })
    }

    fn emulate_missing_fn(
        &mut self,
        path: String,
        _args: &[OpTy<'tcx>],
        dest: Option<PlaceTy<'tcx>>,
        ret: Option<mir::BasicBlock>,
    ) -> EvalResult<'tcx> {
        // In some cases in non-MIR libstd-mode, not having a destination is legit.  Handle these early.
        match &path[..] {
            "std::panicking::rust_panic_with_hook" |
            "core::panicking::panic_fmt::::panic_impl" |
            "std::rt::begin_panic_fmt" =>
                return err!(MachineError("the evaluated program panicked".to_string())),
            _ => {}
        }

        let dest = dest.ok_or_else(
            // Must be some function we do not support
            || EvalErrorKind::NoMirFor(path.clone()),
        )?;

        match &path[..] {
            // A Rust function is missing, which means we are running with MIR missing for libstd (or other dependencies).
            // Still, we can make many things mostly work by "emulating" or ignoring some functions.
            "std::io::_print" |
            "std::io::_eprint" => {
                warn!(
                    "Ignoring output.  To run programs that print, make sure you have a libstd with full MIR."
                );
            }
            "std::thread::Builder::new" => {
                return err!(Unimplemented("miri does not support threading".to_owned()))
            }
            "std::env::args" => {
                return err!(Unimplemented(
                    "miri does not support program arguments".to_owned(),
                ))
            }
            "std::panicking::panicking" |
            "std::rt::panicking" => {
                // we abort on panic -> `std::rt::panicking` always returns false
                self.write_scalar(Scalar::from_bool(false), dest)?;
            }

            _ => return err!(NoMirFor(path)),
        }

        self.goto_block(ret)?;
        self.dump_place(*dest);
        Ok(())
    }

    fn write_null(&mut self, dest: PlaceTy<'tcx>) -> EvalResult<'tcx> {
        self.write_scalar(Scalar::null(dest.layout.size), dest)
    }
}
