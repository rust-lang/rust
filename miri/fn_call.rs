use rustc::ty::{self, Ty};
use rustc::hir::def_id::{DefId, CRATE_DEF_INDEX};
use rustc::mir;
use syntax::attr;
use syntax::abi::Abi;
use syntax::codemap::Span;

use std::mem;

use rustc_miri::interpret::*;

use super::{TlsKey, EvalContext};

use tls::MemoryExt;

use super::memory::MemoryKind;

pub trait EvalContextExt<'tcx> {
    fn call_c_abi(
        &mut self,
        def_id: DefId,
        args: &[ValTy<'tcx>],
        dest: Lvalue,
        dest_ty: Ty<'tcx>,
        dest_block: mir::BasicBlock,
    ) -> EvalResult<'tcx>;

    fn resolve_path(&self, path: &[&str]) -> EvalResult<'tcx, ty::Instance<'tcx>>;

    fn call_missing_fn(
        &mut self,
        instance: ty::Instance<'tcx>,
        destination: Option<(Lvalue, mir::BasicBlock)>,
        args: &[ValTy<'tcx>],
        sig: ty::FnSig<'tcx>,
        path: String,
    ) -> EvalResult<'tcx>;

    fn eval_fn_call(
        &mut self,
        instance: ty::Instance<'tcx>,
        destination: Option<(Lvalue, mir::BasicBlock)>,
        args: &[ValTy<'tcx>],
        span: Span,
        sig: ty::FnSig<'tcx>,
    ) -> EvalResult<'tcx, bool>;
}

impl<'a, 'tcx> EvalContextExt<'tcx> for EvalContext<'a, 'tcx, super::Evaluator> {
    fn eval_fn_call(
        &mut self,
        instance: ty::Instance<'tcx>,
        destination: Option<(Lvalue, mir::BasicBlock)>,
        args: &[ValTy<'tcx>],
        span: Span,
        sig: ty::FnSig<'tcx>,
    ) -> EvalResult<'tcx, bool> {
        trace!("eval_fn_call: {:#?}, {:#?}", instance, destination);

        let mir = match self.load_mir(instance.def) {
            Ok(mir) => mir,
            Err(EvalError { kind: EvalErrorKind::NoMirFor(path), .. }) => {
                self.call_missing_fn(
                    instance,
                    destination,
                    args,
                    sig,
                    path,
                )?;
                return Ok(true);
            }
            Err(other) => return Err(other),
        };

        let (return_lvalue, return_to_block) = match destination {
            Some((lvalue, block)) => (lvalue, StackPopCleanup::Goto(block)),
            None => (Lvalue::undef(), StackPopCleanup::None),
        };

        self.push_stack_frame(
            instance,
            span,
            mir,
            return_lvalue,
            return_to_block,
        )?;

        Ok(false)
    }

    fn call_c_abi(
        &mut self,
        def_id: DefId,
        args: &[ValTy<'tcx>],
        dest: Lvalue,
        dest_ty: Ty<'tcx>,
        dest_block: mir::BasicBlock,
    ) -> EvalResult<'tcx> {
        let name = self.tcx.item_name(def_id);
        let attrs = self.tcx.get_attrs(def_id);
        let link_name = attr::first_attr_value_str_by_name(&attrs, "link_name")
            .unwrap_or(name)
            .as_str();

        match &link_name[..] {
            "malloc" => {
                let size = self.value_to_primval(args[0])?.to_u64()?;
                if size == 0 {
                    self.write_null(dest, dest_ty)?;
                } else {
                    let align = self.memory.pointer_size();
                    let ptr = self.memory.allocate(size, align, MemoryKind::C.into())?;
                    self.write_primval(dest, PrimVal::Ptr(ptr), dest_ty)?;
                }
            }

            "free" => {
                let ptr = args[0].into_ptr(&mut self.memory)?;
                if !ptr.is_null()? {
                    self.memory.deallocate(
                        ptr.to_ptr()?,
                        None,
                        MemoryKind::C.into(),
                    )?;
                }
            }

            "syscall" => {
                // TODO: read `syscall` ids like `sysconf` ids and
                // figure out some way to actually process some of them
                //
                // libc::syscall(NR_GETRANDOM, buf.as_mut_ptr(), buf.len(), GRND_NONBLOCK)
                // is called if a `HashMap` is created the regular way.
                match self.value_to_primval(args[0])?.to_u64()? {
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
                let _handle = args[0].into_ptr(&mut self.memory)?;
                let symbol = args[1].into_ptr(&mut self.memory)?.to_ptr()?;
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

                let arg_local = self.frame().mir.args_iter().next().ok_or(
                    EvalErrorKind::AbiViolation(
                        "Argument to __rust_maybe_catch_panic does not take enough arguments."
                            .to_owned(),
                    ),
                )?;
                let arg_dest = self.eval_lvalue(&mir::Lvalue::Local(arg_local))?;
                self.write_ptr(arg_dest, data, u8_ptr_ty)?;

                // We ourselves return 0
                self.write_null(dest, dest_ty)?;

                // Don't fall through
                return Ok(());
            }

            "__rust_start_panic" => {
                return err!(Panic);
            }

            "memcmp" => {
                let left = args[0].into_ptr(&mut self.memory)?;
                let right = args[1].into_ptr(&mut self.memory)?;
                let n = self.value_to_primval(args[2])?.to_u64()?;

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

                self.write_primval(
                    dest,
                    PrimVal::Bytes(result as u128),
                    dest_ty,
                )?;
            }

            "memrchr" => {
                let ptr = args[0].into_ptr(&mut self.memory)?;
                let val = self.value_to_primval(args[1])?.to_u64()? as u8;
                let num = self.value_to_primval(args[2])?.to_u64()?;
                if let Some(idx) = self.memory.read_bytes(ptr, num)?.iter().rev().position(
                    |&c| c == val,
                )
                {
                    let new_ptr = ptr.offset(num - idx as u64 - 1, &self)?;
                    self.write_ptr(dest, new_ptr, dest_ty)?;
                } else {
                    self.write_null(dest, dest_ty)?;
                }
            }

            "memchr" => {
                let ptr = args[0].into_ptr(&mut self.memory)?;
                let val = self.value_to_primval(args[1])?.to_u64()? as u8;
                let num = self.value_to_primval(args[2])?.to_u64()?;
                if let Some(idx) = self.memory.read_bytes(ptr, num)?.iter().position(
                    |&c| c == val,
                )
                {
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
                        self.memory.deallocate(var, None, MemoryKind::Env.into())?;
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
                    let value_copy = self.memory.allocate(
                        (value.len() + 1) as u64,
                        1,
                        MemoryKind::Env.into(),
                    )?;
                    self.memory.write_bytes(value_copy.into(), &value)?;
                    let trailing_zero_ptr = value_copy.offset(value.len() as u64, &self)?.into();
                    self.memory.write_bytes(trailing_zero_ptr, &[0])?;
                    if let Some(var) = self.machine_data.env_vars.insert(
                        name.to_owned(),
                        value_copy,
                    )
                    {
                        self.memory.deallocate(var, None, MemoryKind::Env.into())?;
                    }
                    self.write_null(dest, dest_ty)?;
                } else {
                    self.write_primval(dest, PrimVal::from_i128(-1), dest_ty)?;
                }
            }

            "write" => {
                let fd = self.value_to_primval(args[0])?.to_u64()?;
                let buf = args[1].into_ptr(&mut self.memory)?;
                let n = self.value_to_primval(args[2])?.to_u64()?;
                trace!("Called write({:?}, {:?}, {:?})", fd, buf, n);
                let result = if fd == 1 || fd == 2 {
                    // stdout/stderr
                    use std::io::{self, Write};

                    let buf_cont = self.memory.read_bytes(buf, n)?;
                    let res = if fd == 1 {
                        io::stdout().write(buf_cont)
                    } else {
                        io::stderr().write(buf_cont)
                    };
                    match res {
                        Ok(n) => n as isize,
                        Err(_) => -1,
                    }
                } else {
                    info!("Ignored output to FD {}", fd);
                    n as isize // pretend it all went well
                }; // now result is the value we return back to the program
                self.write_primval(
                    dest,
                    PrimVal::Bytes(result as u128),
                    dest_ty,
                )?;
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
                let name = self.value_to_primval(args[0])?.to_u64()?;
                trace!("sysconf() called with name {}", name);
                // cache the sysconf integers via miri's global cache
                let paths = &[
                    (&["libc", "_SC_PAGESIZE"], PrimVal::Bytes(4096)),
                    (&["libc", "_SC_GETPW_R_SIZE_MAX"], PrimVal::from_i128(-1)),
                ];
                let mut result = None;
                for &(path, path_value) in paths {
                    if let Ok(instance) = self.resolve_path(path) {
                        let cid = GlobalId {
                            instance,
                            promoted: None,
                        };
                        // compute global if not cached
                        let val = match self.globals.get(&cid).cloned() {
                            Some(ptr) => self.value_to_primval(ValTy { value: Value::ByRef(ptr), ty: args[0].ty })?.to_u64()?,
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
                    return err!(Unimplemented(
                        format!("Unimplemented sysconf name: {}", name),
                    ));
                }
            }

            // Hook pthread calls that go to the thread-local storage memory subsystem
            "pthread_key_create" => {
                let key_ptr = args[0].into_ptr(&mut self.memory)?;

                // Extract the function type out of the signature (that seems easier than constructing it ourselves...)
                let dtor = match args[1].into_ptr(&mut self.memory)?.into_inner_primval() {
                    PrimVal::Ptr(dtor_ptr) => Some(self.memory.get_fn(dtor_ptr)?),
                    PrimVal::Bytes(0) => None,
                    PrimVal::Bytes(_) => return err!(ReadBytesAsPointer),
                    PrimVal::Undef => return err!(ReadUndefBytes),
                };

                // Figure out how large a pthread TLS key actually is. This is libc::pthread_key_t.
                let key_type = args[0].ty.builtin_deref(true, ty::LvaluePreference::NoPreference)
                                   .ok_or(EvalErrorKind::AbiViolation("Wrong signature used for pthread_key_create: First argument must be a raw pointer.".to_owned()))?.ty;
                let key_size = {
                    let layout = self.type_layout(key_type)?;
                    layout.size(&self.tcx.data_layout)
                };

                // Create key and write it into the memory where key_ptr wants it
                let key = self.memory.create_tls_key(dtor) as u128;
                if key_size.bits() < 128 && key >= (1u128 << key_size.bits() as u128) {
                    return err!(OutOfTls);
                }
                self.memory.write_primval(
                    key_ptr.to_ptr()?,
                    PrimVal::Bytes(key),
                    key_size.bytes(),
                    false,
                )?;

                // Return success (0)
                self.write_null(dest, dest_ty)?;
            }
            "pthread_key_delete" => {
                // The conversion into TlsKey here is a little fishy, but should work as long as usize >= libc::pthread_key_t
                let key = self.value_to_primval(args[0])?.to_u64()? as TlsKey;
                self.memory.delete_tls_key(key)?;
                // Return success (0)
                self.write_null(dest, dest_ty)?;
            }
            "pthread_getspecific" => {
                // The conversion into TlsKey here is a little fishy, but should work as long as usize >= libc::pthread_key_t
                let key = self.value_to_primval(args[0])?.to_u64()? as TlsKey;
                let ptr = self.memory.load_tls(key)?;
                self.write_ptr(dest, ptr, dest_ty)?;
            }
            "pthread_setspecific" => {
                // The conversion into TlsKey here is a little fishy, but should work as long as usize >= libc::pthread_key_t
                let key = self.value_to_primval(args[0])?.to_u64()? as TlsKey;
                let new_ptr = args[1].into_ptr(&mut self.memory)?;
                self.memory.store_tls(key, new_ptr)?;

                // Return success (0)
                self.write_null(dest, dest_ty)?;
            }

            // Stub out all the other pthread calls to just return 0
            link_name if link_name.starts_with("pthread_") => {
                warn!("ignoring C ABI call: {}", link_name);
                self.write_null(dest, dest_ty)?;
            }

            _ => {
                return err!(Unimplemented(
                    format!("can't call C ABI function: {}", link_name),
                ));
            }
        }

        // Since we pushed no stack frame, the main loop will act
        // as if the call just completed and it's returning to the
        // current frame.
        self.dump_local(dest);
        self.goto_block(dest_block);
        Ok(())
    }

    /// Get an instance for a path.
    fn resolve_path(&self, path: &[&str]) -> EvalResult<'tcx, ty::Instance<'tcx>> {
        let cstore = &self.tcx.sess.cstore;

        let crates = cstore.crates();
        crates
            .iter()
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
                let path = path.iter().map(|&s| s.to_owned()).collect();
                EvalErrorKind::PathNotFound(path).into()
            })
    }

    fn call_missing_fn(
        &mut self,
        instance: ty::Instance<'tcx>,
        destination: Option<(Lvalue, mir::BasicBlock)>,
        args: &[ValTy<'tcx>],
        sig: ty::FnSig<'tcx>,
        path: String,
    ) -> EvalResult<'tcx> {
        // In some cases in non-MIR libstd-mode, not having a destination is legit.  Handle these early.
        match &path[..] {
            "std::panicking::rust_panic_with_hook" |
            "std::rt::begin_panic_fmt" => return err!(Panic),
            _ => {}
        }

        let dest_ty = sig.output();
        let (dest, dest_block) = destination.ok_or_else(
            || EvalErrorKind::NoMirFor(path.clone()),
        )?;

        if sig.abi == Abi::C {
            // An external C function
            // TODO: That functions actually has a similar preamble to what follows here.  May make sense to
            // unify these two mechanisms for "hooking into missing functions".
            self.call_c_abi(
                instance.def_id(),
                args,
                dest,
                dest_ty,
                dest_block,
            )?;
            return Ok(());
        }

        match &path[..] {
            // Allocators are magic.  They have no MIR, even when the rest of libstd does.
            "alloc::heap::::__rust_alloc" => {
                let size = self.value_to_primval(args[0])?.to_u64()?;
                let align = self.value_to_primval(args[1])?.to_u64()?;
                if size == 0 {
                    return err!(HeapAllocZeroBytes);
                }
                if !align.is_power_of_two() {
                    return err!(HeapAllocNonPowerOfTwoAlignment(align));
                }
                let ptr = self.memory.allocate(size, align, MemoryKind::Rust.into())?;
                self.write_primval(dest, PrimVal::Ptr(ptr), dest_ty)?;
            }
            "alloc::heap::::__rust_alloc_zeroed" => {
                let size = self.value_to_primval(args[0])?.to_u64()?;
                let align = self.value_to_primval(args[1])?.to_u64()?;
                if size == 0 {
                    return err!(HeapAllocZeroBytes);
                }
                if !align.is_power_of_two() {
                    return err!(HeapAllocNonPowerOfTwoAlignment(align));
                }
                let ptr = self.memory.allocate(size, align, MemoryKind::Rust.into())?;
                self.memory.write_repeat(ptr.into(), 0, size)?;
                self.write_primval(dest, PrimVal::Ptr(ptr), dest_ty)?;
            }
            "alloc::heap::::__rust_dealloc" => {
                let ptr = args[0].into_ptr(&mut self.memory)?.to_ptr()?;
                let old_size = self.value_to_primval(args[1])?.to_u64()?;
                let align = self.value_to_primval(args[2])?.to_u64()?;
                if old_size == 0 {
                    return err!(HeapAllocZeroBytes);
                }
                if !align.is_power_of_two() {
                    return err!(HeapAllocNonPowerOfTwoAlignment(align));
                }
                self.memory.deallocate(
                    ptr,
                    Some((old_size, align)),
                    MemoryKind::Rust.into(),
                )?;
            }
            "alloc::heap::::__rust_realloc" => {
                let ptr = args[0].into_ptr(&mut self.memory)?.to_ptr()?;
                let old_size = self.value_to_primval(args[1])?.to_u64()?;
                let old_align = self.value_to_primval(args[2])?.to_u64()?;
                let new_size = self.value_to_primval(args[3])?.to_u64()?;
                let new_align = self.value_to_primval(args[4])?.to_u64()?;
                if old_size == 0 || new_size == 0 {
                    return err!(HeapAllocZeroBytes);
                }
                if !old_align.is_power_of_two() {
                    return err!(HeapAllocNonPowerOfTwoAlignment(old_align));
                }
                if !new_align.is_power_of_two() {
                    return err!(HeapAllocNonPowerOfTwoAlignment(new_align));
                }
                let new_ptr = self.memory.reallocate(
                    ptr,
                    old_size,
                    old_align,
                    new_size,
                    new_align,
                    MemoryKind::Rust.into(),
                )?;
                self.write_primval(dest, PrimVal::Ptr(new_ptr), dest_ty)?;
            }

            // A Rust function is missing, which means we are running with MIR missing for libstd (or other dependencies).
            // Still, we can make many things mostly work by "emulating" or ignoring some functions.
            "std::io::_print" => {
                trace!(
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
                let bool = self.tcx.types.bool;
                self.write_primval(dest, PrimVal::from_bool(false), bool)?;
            }
            _ => return err!(NoMirFor(path)),
        }

        // Since we pushed no stack frame, the main loop will act
        // as if the call just completed and it's returning to the
        // current frame.
        self.dump_local(dest);
        self.goto_block(dest_block);
        return Ok(());
    }
}
