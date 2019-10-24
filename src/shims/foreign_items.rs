use std::{iter, convert::TryInto};

use rustc::hir::def_id::DefId;
use rustc::mir;
use rustc::ty::layout::{Align, LayoutOf, Size};
use rustc_apfloat::Float;
use syntax::attr;
use syntax::symbol::sym;

use crate::*;

impl<'mir, 'tcx> EvalContextExt<'mir, 'tcx> for crate::MiriEvalContext<'mir, 'tcx> {}
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriEvalContextExt<'mir, 'tcx> {
    /// Returns the minimum alignment for the target architecture for allocations of the given size.
    fn min_align(&self, size: u64, kind: MiriMemoryKind) -> Align {
        let this = self.eval_context_ref();
        // List taken from `libstd/sys_common/alloc.rs`.
        let min_align = match this.tcx.tcx.sess.target.target.arch.as_str() {
            "x86" | "arm" | "mips" | "powerpc" | "powerpc64" | "asmjs" | "wasm32" => 8,
            "x86_64" | "aarch64" | "mips64" | "s390x" | "sparc64" => 16,
            arch => bug!("Unsupported target architecture: {}", arch),
        };
        // Windows always aligns, even small allocations.
        // Source: <https://support.microsoft.com/en-us/help/286470/how-to-use-pageheap-exe-in-windows-xp-windows-2000-and-windows-server>
        // But jemalloc does not, so for the C heap we only align if the allocation is sufficiently big.
        if kind == MiriMemoryKind::WinHeap || size >= min_align {
            return Align::from_bytes(min_align).unwrap();
        }
        // We have `size < min_align`. Round `size` *down* to the next power of two and use that.
        fn prev_power_of_two(x: u64) -> u64 {
            let next_pow2 = x.next_power_of_two();
            if next_pow2 == x {
                // x *is* a power of two, just use that.
                x
            } else {
                // x is between two powers, so next = 2*prev.
                next_pow2 / 2
            }
        }
        Align::from_bytes(prev_power_of_two(size)).unwrap()
    }

    fn malloc(&mut self, size: u64, zero_init: bool, kind: MiriMemoryKind) -> Scalar<Tag> {
        let this = self.eval_context_mut();
        if size == 0 {
            Scalar::from_int(0, this.pointer_size())
        } else {
            let align = this.min_align(size, kind);
            let ptr = this
                .memory
                .allocate(Size::from_bytes(size), align, kind.into());
            if zero_init {
                // We just allocated this, the access is definitely in-bounds.
                this.memory
                    .write_bytes(ptr.into(), iter::repeat(0u8).take(size as usize))
                    .unwrap();
            }
            Scalar::Ptr(ptr)
        }
    }

    fn free(&mut self, ptr: Scalar<Tag>, kind: MiriMemoryKind) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        if !this.is_null(ptr)? {
            let ptr = this.force_ptr(ptr)?;
            this.memory.deallocate(ptr, None, kind.into())?;
        }
        Ok(())
    }

    fn realloc(
        &mut self,
        old_ptr: Scalar<Tag>,
        new_size: u64,
        kind: MiriMemoryKind,
    ) -> InterpResult<'tcx, Scalar<Tag>> {
        let this = self.eval_context_mut();
        let new_align = this.min_align(new_size, kind);
        if this.is_null(old_ptr)? {
            if new_size == 0 {
                Ok(Scalar::from_int(0, this.pointer_size()))
            } else {
                let new_ptr =
                    this.memory
                        .allocate(Size::from_bytes(new_size), new_align, kind.into());
                Ok(Scalar::Ptr(new_ptr))
            }
        } else {
            let old_ptr = this.force_ptr(old_ptr)?;
            if new_size == 0 {
                this.memory.deallocate(old_ptr, None, kind.into())?;
                Ok(Scalar::from_int(0, this.pointer_size()))
            } else {
                let new_ptr = this.memory.reallocate(
                    old_ptr,
                    None,
                    Size::from_bytes(new_size),
                    new_align,
                    kind.into(),
                )?;
                Ok(Scalar::Ptr(new_ptr))
            }
        }
    }

    /// Emulates calling a foreign item, failing if the item is not supported.
    /// This function will handle `goto_block` if needed.
    fn emulate_foreign_item(
        &mut self,
        def_id: DefId,
        args: &[OpTy<'tcx, Tag>],
        dest: Option<PlaceTy<'tcx, Tag>>,
        ret: Option<mir::BasicBlock>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        let attrs = this.tcx.get_attrs(def_id);
        let link_name = match attr::first_attr_value_str_by_name(&attrs, sym::link_name) {
            Some(name) => name.as_str(),
            None => this.tcx.item_name(def_id).as_str(),
        };
        // Strip linker suffixes (seen on 32-bit macOS).
        let link_name = link_name.trim_end_matches("$UNIX2003");
        let tcx = &{ this.tcx.tcx };

        // First: functions that diverge.
        match link_name {
            "__rust_start_panic" | "panic_impl" => {
                throw_unsup_format!("the evaluated program panicked");
            }
            "exit" | "ExitProcess" => {
                // it's really u32 for ExitProcess, but we have to put it into the `Exit` error variant anyway
                let code = this.read_scalar(args[0])?.to_i32()?;
                return Err(InterpError::Exit(code).into());
            }
            _ => {
                if dest.is_none() {
                    throw_unsup_format!("can't call (diverging) foreign function: {}", link_name);
                }
            }
        }

        // Next: functions that assume a ret and dest.
        let dest = dest.expect("we already checked for a dest");
        let ret = ret.expect("dest is `Some` but ret is `None`");
        match link_name {
            "malloc" => {
                let size = this.read_scalar(args[0])?.to_usize(this)?;
                let res = this.malloc(size, /*zero_init:*/ false, MiriMemoryKind::C);
                this.write_scalar(res, dest)?;
            }
            "calloc" => {
                let items = this.read_scalar(args[0])?.to_usize(this)?;
                let len = this.read_scalar(args[1])?.to_usize(this)?;
                let size = items
                    .checked_mul(len)
                    .ok_or_else(|| err_panic!(Overflow(mir::BinOp::Mul)))?;
                let res = this.malloc(size, /*zero_init:*/ true, MiriMemoryKind::C);
                this.write_scalar(res, dest)?;
            }
            "posix_memalign" => {
                let ret = this.deref_operand(args[0])?;
                let align = this.read_scalar(args[1])?.to_usize(this)?;
                let size = this.read_scalar(args[2])?.to_usize(this)?;
                // Align must be power of 2, and also at least ptr-sized (POSIX rules).
                if !align.is_power_of_two() {
                    throw_unsup!(HeapAllocNonPowerOfTwoAlignment(align));
                }
                if align < this.pointer_size().bytes() {
                    throw_ub_format!(
                        "posix_memalign: alignment must be at least the size of a pointer, but is {}",
                        align,
                    );
                }

                if size == 0 {
                    this.write_null(ret.into())?;
                } else {
                    let ptr = this.memory.allocate(
                        Size::from_bytes(size),
                        Align::from_bytes(align).unwrap(),
                        MiriMemoryKind::C.into(),
                    );
                    this.write_scalar(Scalar::Ptr(ptr), ret.into())?;
                }
                this.write_null(dest)?;
            }
            "free" => {
                let ptr = this.read_scalar(args[0])?.not_undef()?;
                this.free(ptr, MiriMemoryKind::C)?;
            }
            "realloc" => {
                let old_ptr = this.read_scalar(args[0])?.not_undef()?;
                let new_size = this.read_scalar(args[1])?.to_usize(this)?;
                let res = this.realloc(old_ptr, new_size, MiriMemoryKind::C)?;
                this.write_scalar(res, dest)?;
            }

            "__rust_alloc" => {
                let size = this.read_scalar(args[0])?.to_usize(this)?;
                let align = this.read_scalar(args[1])?.to_usize(this)?;
                if size == 0 {
                    throw_unsup!(HeapAllocZeroBytes);
                }
                if !align.is_power_of_two() {
                    throw_unsup!(HeapAllocNonPowerOfTwoAlignment(align));
                }
                let ptr = this.memory.allocate(
                    Size::from_bytes(size),
                    Align::from_bytes(align).unwrap(),
                    MiriMemoryKind::Rust.into(),
                );
                this.write_scalar(Scalar::Ptr(ptr), dest)?;
            }
            "__rust_alloc_zeroed" => {
                let size = this.read_scalar(args[0])?.to_usize(this)?;
                let align = this.read_scalar(args[1])?.to_usize(this)?;
                if size == 0 {
                    throw_unsup!(HeapAllocZeroBytes);
                }
                if !align.is_power_of_two() {
                    throw_unsup!(HeapAllocNonPowerOfTwoAlignment(align));
                }
                let ptr = this.memory.allocate(
                    Size::from_bytes(size),
                    Align::from_bytes(align).unwrap(),
                    MiriMemoryKind::Rust.into(),
                );
                // We just allocated this, the access is definitely in-bounds.
                this.memory
                    .write_bytes(ptr.into(), iter::repeat(0u8).take(size as usize))
                    .unwrap();
                this.write_scalar(Scalar::Ptr(ptr), dest)?;
            }
            "__rust_dealloc" => {
                let ptr = this.read_scalar(args[0])?.not_undef()?;
                let old_size = this.read_scalar(args[1])?.to_usize(this)?;
                let align = this.read_scalar(args[2])?.to_usize(this)?;
                if old_size == 0 {
                    throw_unsup!(HeapAllocZeroBytes);
                }
                if !align.is_power_of_two() {
                    throw_unsup!(HeapAllocNonPowerOfTwoAlignment(align));
                }
                let ptr = this.force_ptr(ptr)?;
                this.memory.deallocate(
                    ptr,
                    Some((
                        Size::from_bytes(old_size),
                        Align::from_bytes(align).unwrap(),
                    )),
                    MiriMemoryKind::Rust.into(),
                )?;
            }
            "__rust_realloc" => {
                let ptr = this.read_scalar(args[0])?.to_ptr()?;
                let old_size = this.read_scalar(args[1])?.to_usize(this)?;
                let align = this.read_scalar(args[2])?.to_usize(this)?;
                let new_size = this.read_scalar(args[3])?.to_usize(this)?;
                if old_size == 0 || new_size == 0 {
                    throw_unsup!(HeapAllocZeroBytes);
                }
                if !align.is_power_of_two() {
                    throw_unsup!(HeapAllocNonPowerOfTwoAlignment(align));
                }
                let align = Align::from_bytes(align).unwrap();
                let new_ptr = this.memory.reallocate(
                    ptr,
                    Some((Size::from_bytes(old_size), align)),
                    Size::from_bytes(new_size),
                    align,
                    MiriMemoryKind::Rust.into(),
                )?;
                this.write_scalar(Scalar::Ptr(new_ptr), dest)?;
            }

            "syscall" => {
                let sys_getrandom = this
                    .eval_path_scalar(&["libc", "SYS_getrandom"])?
                    .expect("Failed to get libc::SYS_getrandom")
                    .to_usize(this)?;

                // `libc::syscall(NR_GETRANDOM, buf.as_mut_ptr(), buf.len(), GRND_NONBLOCK)`
                // is called if a `HashMap` is created the regular way (e.g. HashMap<K, V>).
                match this.read_scalar(args[0])?.to_usize(this)? {
                    id if id == sys_getrandom => {
                        // The first argument is the syscall id,
                        // so skip over it.
                        linux_getrandom(this, &args[1..], dest)?;
                    }
                    id => throw_unsup_format!("miri does not support syscall ID {}", id),
                }
            }

            "getrandom" => {
                linux_getrandom(this, args, dest)?;
            }

            "dlsym" => {
                let _handle = this.read_scalar(args[0])?;
                let symbol = this.read_scalar(args[1])?.not_undef()?;
                let symbol_name = this.memory.read_c_str(symbol)?;
                let err = format!("bad c unicode symbol: {:?}", symbol_name);
                let symbol_name = ::std::str::from_utf8(symbol_name).unwrap_or(&err);
                if let Some(dlsym) = Dlsym::from_str(symbol_name)? {
                    let ptr = this.memory.create_fn_alloc(FnVal::Other(dlsym));
                    this.write_scalar(Scalar::from(ptr), dest)?;
                } else {
                    this.write_null(dest)?;
                }
            }

            "__rust_maybe_catch_panic" => {
                // fn __rust_maybe_catch_panic(
                //     f: fn(*mut u8),
                //     data: *mut u8,
                //     data_ptr: *mut usize,
                //     vtable_ptr: *mut usize,
                // ) -> u32
                // We abort on panic, so not much is going on here, but we still have to call the closure.
                let f = this.read_scalar(args[0])?.not_undef()?;
                let data = this.read_scalar(args[1])?.not_undef()?;
                let f_instance = this.memory.get_fn(f)?.as_instance()?;
                this.write_null(dest)?;
                trace!("__rust_maybe_catch_panic: {:?}", f_instance);

                // Now we make a function call.
                // TODO: consider making this reusable? `InterpCx::step` does something similar
                // for the TLS destructors, and of course `eval_main`.
                let mir = this.load_mir(f_instance.def, None)?;
                let ret_place =
                    MPlaceTy::dangling(this.layout_of(tcx.mk_unit())?, this).into();
                this.push_stack_frame(
                    f_instance,
                    mir.span,
                    mir,
                    Some(ret_place),
                    // Directly return to caller.
                    StackPopCleanup::Goto(Some(ret)),
                )?;
                let mut args = this.frame().body.args_iter();

                let arg_local = args
                    .next()
                    .expect("Argument to __rust_maybe_catch_panic does not take enough arguments.");
                let arg_dest = this.local_place(arg_local)?;
                this.write_scalar(data, arg_dest)?;

                args.next().expect_none("__rust_maybe_catch_panic argument has more arguments than expected");

                // We ourselves will return `0`, eventually (because we will not return if we paniced).
                this.write_null(dest)?;

                // Don't fall through, we do *not* want to `goto_block`!
                return Ok(());
            }

            "memcmp" => {
                let left = this.read_scalar(args[0])?.not_undef()?;
                let right = this.read_scalar(args[1])?.not_undef()?;
                let n = Size::from_bytes(this.read_scalar(args[2])?.to_usize(this)?);

                let result = {
                    let left_bytes = this.memory.read_bytes(left, n)?;
                    let right_bytes = this.memory.read_bytes(right, n)?;

                    use std::cmp::Ordering::*;
                    match left_bytes.cmp(right_bytes) {
                        Less => -1i32,
                        Equal => 0,
                        Greater => 1,
                    }
                };

                this.write_scalar(Scalar::from_int(result, Size::from_bits(32)), dest)?;
            }

            "memrchr" => {
                let ptr = this.read_scalar(args[0])?.not_undef()?;
                let val = this.read_scalar(args[1])?.to_i32()? as u8;
                let num = this.read_scalar(args[2])?.to_usize(this)?;
                if let Some(idx) = this
                    .memory
                    .read_bytes(ptr, Size::from_bytes(num))?
                    .iter()
                    .rev()
                    .position(|&c| c == val)
                {
                    let new_ptr = ptr.ptr_offset(Size::from_bytes(num - idx as u64 - 1), this)?;
                    this.write_scalar(new_ptr, dest)?;
                } else {
                    this.write_null(dest)?;
                }
            }

            "memchr" => {
                let ptr = this.read_scalar(args[0])?.not_undef()?;
                let val = this.read_scalar(args[1])?.to_i32()? as u8;
                let num = this.read_scalar(args[2])?.to_usize(this)?;
                let idx = this
                    .memory
                    .read_bytes(ptr, Size::from_bytes(num))?
                    .iter()
                    .position(|&c| c == val);
                if let Some(idx) = idx {
                    let new_ptr = ptr.ptr_offset(Size::from_bytes(idx as u64), this)?;
                    this.write_scalar(new_ptr, dest)?;
                } else {
                    this.write_null(dest)?;
                }
            }

            "__errno_location" | "__error" => {
                let errno_place = this.machine.last_error.unwrap();
                this.write_scalar(errno_place.to_ref().to_scalar()?, dest)?;
            }

            "getenv" => {
                let result = this.getenv(args[0])?;
                this.write_scalar(result, dest)?;
            }

            "unsetenv" => {
                let result = this.unsetenv(args[0])?;
                this.write_scalar(Scalar::from_int(result, dest.layout.size), dest)?;
            }

            "setenv" => {
                let result = this.setenv(args[0], args[1])?;
                this.write_scalar(Scalar::from_int(result, dest.layout.size), dest)?;
            }

            "getcwd" => {
                let result = this.getcwd(args[0], args[1])?;
                this.write_scalar(result, dest)?;
            }

            "chdir" => {
                let result = this.chdir(args[0])?;
                this.write_scalar(Scalar::from_int(result, dest.layout.size), dest)?;
            }

            "open" | "open64" => {
                let result = this.open(args[0], args[1])?;
                this.write_scalar(Scalar::from_int(result, dest.layout.size), dest)?;
            }

            "fcntl" => {
                let result = this.fcntl(args[0], args[1], args.get(2).cloned())?;
                this.write_scalar(Scalar::from_int(result, dest.layout.size), dest)?;
            }

            "close" | "close$NOCANCEL" => {
                let result = this.close(args[0])?;
                this.write_scalar(Scalar::from_int(result, dest.layout.size), dest)?;
            }

            "read" => {
                let result = this.read(args[0], args[1], args[2])?;
                this.write_scalar(Scalar::from_int(result, dest.layout.size), dest)?;
            }

            "write" => {
                let fd = this.read_scalar(args[0])?.to_i32()?;
                let buf = this.read_scalar(args[1])?.not_undef()?;
                let n = this.read_scalar(args[2])?.to_usize(tcx)?;
                trace!("Called write({:?}, {:?}, {:?})", fd, buf, n);
                let result = if fd == 1 || fd == 2 {
                    // stdout/stderr
                    use std::io::{self, Write};

                    let buf_cont = this.memory.read_bytes(buf, Size::from_bytes(n))?;
                    // We need to flush to make sure this actually appears on the screen
                    let res = if fd == 1 {
                        // Stdout is buffered, flush to make sure it appears on the screen.
                        // This is the write() syscall of the interpreted program, we want it
                        // to correspond to a write() syscall on the host -- there is no good
                        // in adding extra buffering here.
                        let res = io::stdout().write(buf_cont);
                        io::stdout().flush().unwrap();
                        res
                    } else {
                        // No need to flush, stderr is not buffered.
                        io::stderr().write(buf_cont)
                    };
                    match res {
                        Ok(n) => n as i64,
                        Err(_) => -1,
                    }
                } else {
                    this.write(args[0], args[1], args[2])?
                };
                // Now, `result` is the value we return back to the program.
                this.write_scalar(Scalar::from_int(result, dest.layout.size), dest)?;
            }

            "unlink" => {
                let result = this.unlink(args[0])?;
                this.write_scalar(Scalar::from_int(result, dest.layout.size), dest)?;
            }

            "clock_gettime" => {
                let result = this.clock_gettime(args[0], args[1])?;
                this.write_scalar(Scalar::from_int(result, dest.layout.size), dest)?;
            }

            "gettimeofday" => {
                let result = this.gettimeofday(args[0], args[1])?;
                this.write_scalar(Scalar::from_int(result, dest.layout.size), dest)?;
            }

            "strlen" => {
                let ptr = this.read_scalar(args[0])?.not_undef()?;
                let n = this.memory.read_c_str(ptr)?.len();
                this.write_scalar(Scalar::from_uint(n as u64, dest.layout.size), dest)?;
            }

            // math functions
            "cbrtf" | "coshf" | "sinhf" | "tanf" => {
                // FIXME: Using host floats.
                let f = f32::from_bits(this.read_scalar(args[0])?.to_u32()?);
                let f = match link_name {
                    "cbrtf" => f.cbrt(),
                    "coshf" => f.cosh(),
                    "sinhf" => f.sinh(),
                    "tanf" => f.tan(),
                    _ => bug!(),
                };
                this.write_scalar(Scalar::from_u32(f.to_bits()), dest)?;
            }
            // underscore case for windows
            "_hypotf" | "hypotf" | "atan2f" => {
                // FIXME: Using host floats.
                let f1 = f32::from_bits(this.read_scalar(args[0])?.to_u32()?);
                let f2 = f32::from_bits(this.read_scalar(args[1])?.to_u32()?);
                let n = match link_name {
                    "_hypotf" | "hypotf" => f1.hypot(f2),
                    "atan2f" => f1.atan2(f2),
                    _ => bug!(),
                };
                this.write_scalar(Scalar::from_u32(n.to_bits()), dest)?;
            }

            "cbrt" | "cosh" | "sinh" | "tan" => {
                // FIXME: Using host floats.
                let f = f64::from_bits(this.read_scalar(args[0])?.to_u64()?);
                let f = match link_name {
                    "cbrt" => f.cbrt(),
                    "cosh" => f.cosh(),
                    "sinh" => f.sinh(),
                    "tan" => f.tan(),
                    _ => bug!(),
                };
                this.write_scalar(Scalar::from_u64(f.to_bits()), dest)?;
            }
            // underscore case for windows, here and below
            // (see https://docs.microsoft.com/en-us/cpp/c-runtime-library/reference/floating-point-primitives?view=vs-2019)
            "_hypot" | "hypot" | "atan2" => {
                // FIXME: Using host floats.
                let f1 = f64::from_bits(this.read_scalar(args[0])?.to_u64()?);
                let f2 = f64::from_bits(this.read_scalar(args[1])?.to_u64()?);
                let n = match link_name {
                    "_hypot" | "hypot" => f1.hypot(f2),
                    "atan2" => f1.atan2(f2),
                    _ => bug!(),
                };
                this.write_scalar(Scalar::from_u64(n.to_bits()), dest)?;
            }
            // For radix-2 (binary) systems, `ldexp` and `scalbn` are the same.
            "_ldexp" | "ldexp" | "scalbn" => {
                let x = this.read_scalar(args[0])?.to_f64()?;
                let exp = this.read_scalar(args[1])?.to_i32()?;

                // Saturating cast to i16. Even those are outside the valid exponent range to
                // `scalbn` below will do its over/underflow handling.
                let exp = if exp > i16::max_value() as i32 {
                    i16::max_value()
                } else if exp < i16::min_value() as i32 {
                    i16::min_value()
                } else {
                    exp.try_into().unwrap()
                };

                let res = x.scalbn(exp);
                this.write_scalar(Scalar::from_f64(res), dest)?;
            }

            // Some things needed for `sys::thread` initialization to go through.
            "signal" | "sigaction" | "sigaltstack" => {
                this.write_scalar(Scalar::from_int(0, dest.layout.size), dest)?;
            }

            "sysconf" => {
                let name = this.read_scalar(args[0])?.to_i32()?;

                trace!("sysconf() called with name {}", name);
                // TODO: Cache the sysconf integers via Miri's global cache.
                let paths = &[
                    (
                        &["libc", "_SC_PAGESIZE"],
                        Scalar::from_int(PAGE_SIZE, dest.layout.size),
                    ),
                    (
                        &["libc", "_SC_GETPW_R_SIZE_MAX"],
                        Scalar::from_int(-1, dest.layout.size),
                    ),
                    (
                        &["libc", "_SC_NPROCESSORS_ONLN"],
                        Scalar::from_int(NUM_CPUS, dest.layout.size),
                    ),
                ];
                let mut result = None;
                for &(path, path_value) in paths {
                    if let Some(val) = this.eval_path_scalar(path)? {
                        let val = val.to_i32()?;
                        if val == name {
                            result = Some(path_value);
                            break;
                        }
                    }
                }
                if let Some(result) = result {
                    this.write_scalar(result, dest)?;
                } else {
                    throw_unsup_format!("Unimplemented sysconf name: {}", name)
                }
            }

            "sched_getaffinity" => {
                // Return an error; `num_cpus` then falls back to `sysconf`.
                this.write_scalar(Scalar::from_int(-1, dest.layout.size), dest)?;
            }

            "isatty" => {
                this.write_null(dest)?;
            }

            // Hook pthread calls that go to the thread-local storage memory subsystem.
            "pthread_key_create" => {
                let key_place = this.deref_operand(args[0])?;

                // Extract the function type out of the signature (that seems easier than constructing it ourselves).
                let dtor = match this.test_null(this.read_scalar(args[1])?.not_undef()?)? {
                    Some(dtor_ptr) => Some(this.memory.get_fn(dtor_ptr)?.as_instance()?),
                    None => None,
                };

                // Figure out how large a pthread TLS key actually is.
                // This is `libc::pthread_key_t`.
                let key_type = args[0].layout.ty
                    .builtin_deref(true)
                    .ok_or_else(|| err_ub_format!(
                        "wrong signature used for `pthread_key_create`: first argument must be a raw pointer."
                    ))?
                    .ty;
                let key_layout = this.layout_of(key_type)?;

                // Create key and write it into the memory where `key_ptr` wants it.
                let key = this.machine.tls.create_tls_key(dtor) as u128;
                if key_layout.size.bits() < 128 && key >= (1u128 << key_layout.size.bits() as u128)
                {
                    throw_unsup!(OutOfTls);
                }

                this.write_scalar(Scalar::from_uint(key, key_layout.size), key_place.into())?;

                // Return success (`0`).
                this.write_null(dest)?;
            }
            "pthread_key_delete" => {
                let key = this.read_scalar(args[0])?.to_bits(args[0].layout.size)?;
                this.machine.tls.delete_tls_key(key)?;
                // Return success (0)
                this.write_null(dest)?;
            }
            "pthread_getspecific" => {
                let key = this.read_scalar(args[0])?.to_bits(args[0].layout.size)?;
                let ptr = this.machine.tls.load_tls(key, tcx)?;
                this.write_scalar(ptr, dest)?;
            }
            "pthread_setspecific" => {
                let key = this.read_scalar(args[0])?.to_bits(args[0].layout.size)?;
                let new_ptr = this.read_scalar(args[1])?.not_undef()?;
                this.machine.tls.store_tls(key, this.test_null(new_ptr)?)?;

                // Return success (`0`).
                this.write_null(dest)?;
            }

            // Stack size/address stuff.
            "pthread_attr_init"
            | "pthread_attr_destroy"
            | "pthread_self"
            | "pthread_attr_setstacksize" => {
                this.write_null(dest)?;
            }
            "pthread_attr_getstack" => {
                let addr_place = this.deref_operand(args[1])?;
                let size_place = this.deref_operand(args[2])?;

                this.write_scalar(
                    Scalar::from_uint(STACK_ADDR, addr_place.layout.size),
                    addr_place.into(),
                )?;
                this.write_scalar(
                    Scalar::from_uint(STACK_SIZE, size_place.layout.size),
                    size_place.into(),
                )?;

                // Return success (`0`).
                this.write_null(dest)?;
            }

            // We don't support threading. (Also for Windows.)
            "pthread_create" | "CreateThread" => {
                throw_unsup_format!("Miri does not support threading");
            }

            // Stub out calls for condvar, mutex and rwlock, to just return `0`.
            "pthread_mutexattr_init"
            | "pthread_mutexattr_settype"
            | "pthread_mutex_init"
            | "pthread_mutexattr_destroy"
            | "pthread_mutex_lock"
            | "pthread_mutex_unlock"
            | "pthread_mutex_destroy"
            | "pthread_rwlock_rdlock"
            | "pthread_rwlock_unlock"
            | "pthread_rwlock_wrlock"
            | "pthread_rwlock_destroy"
            | "pthread_condattr_init"
            | "pthread_condattr_setclock"
            | "pthread_cond_init"
            | "pthread_condattr_destroy"
            | "pthread_cond_destroy" => {
                this.write_null(dest)?;
            }

            // We don't support fork so we don't have to do anything for atfork.
            "pthread_atfork" => {
                this.write_null(dest)?;
            }

            "mmap" => {
                // This is a horrible hack, but since the guard page mechanism calls mmap and expects a particular return value, we just give it that value.
                let addr = this.read_scalar(args[0])?.not_undef()?;
                this.write_scalar(addr, dest)?;
            }
            "mprotect" => {
                this.write_null(dest)?;
            }

            // macOS API stubs.
            "pthread_attr_get_np" | "pthread_getattr_np" => {
                this.write_null(dest)?;
            }
            "pthread_get_stackaddr_np" => {
                let stack_addr = Scalar::from_uint(STACK_ADDR, dest.layout.size);
                this.write_scalar(stack_addr, dest)?;
            }
            "pthread_get_stacksize_np" => {
                let stack_size = Scalar::from_uint(STACK_SIZE, dest.layout.size);
                this.write_scalar(stack_size, dest)?;
            }
            "_tlv_atexit" => {
                // FIXME: register the destructor.
            }
            "_NSGetArgc" => {
                this.write_scalar(Scalar::Ptr(this.machine.argc.unwrap()), dest)?;
            }
            "_NSGetArgv" => {
                this.write_scalar(Scalar::Ptr(this.machine.argv.unwrap()), dest)?;
            }
            "SecRandomCopyBytes" => {
                let len = this.read_scalar(args[1])?.to_usize(this)?;
                let ptr = this.read_scalar(args[2])?.not_undef()?;
                this.gen_random(ptr, len as usize)?;
                this.write_null(dest)?;
            }

            // Windows API stubs.
            // HANDLE = isize
            // DWORD = ULONG = u32
            // BOOL = i32
            "GetProcessHeap" => {
                // Just fake a HANDLE
                this.write_scalar(Scalar::from_int(1, this.pointer_size()), dest)?;
            }
            "HeapAlloc" => {
                let _handle = this.read_scalar(args[0])?.to_isize(this)?;
                let flags = this.read_scalar(args[1])?.to_u32()?;
                let size = this.read_scalar(args[2])?.to_usize(this)?;
                let zero_init = (flags & 0x00000008) != 0; // HEAP_ZERO_MEMORY
                let res = this.malloc(size, zero_init, MiriMemoryKind::WinHeap);
                this.write_scalar(res, dest)?;
            }
            "HeapFree" => {
                let _handle = this.read_scalar(args[0])?.to_isize(this)?;
                let _flags = this.read_scalar(args[1])?.to_u32()?;
                let ptr = this.read_scalar(args[2])?.not_undef()?;
                this.free(ptr, MiriMemoryKind::WinHeap)?;
                this.write_scalar(Scalar::from_int(1, Size::from_bytes(4)), dest)?;
            }
            "HeapReAlloc" => {
                let _handle = this.read_scalar(args[0])?.to_isize(this)?;
                let _flags = this.read_scalar(args[1])?.to_u32()?;
                let ptr = this.read_scalar(args[2])?.not_undef()?;
                let size = this.read_scalar(args[3])?.to_usize(this)?;
                let res = this.realloc(ptr, size, MiriMemoryKind::WinHeap)?;
                this.write_scalar(res, dest)?;
            }

            "SetLastError" => {
                this.set_last_error(this.read_scalar(args[0])?.not_undef()?)?;
            }
            "GetLastError" => {
                let last_error = this.get_last_error()?;
                this.write_scalar(last_error, dest)?;
            }

            "AddVectoredExceptionHandler" => {
                // Any non zero value works for the stdlib. This is just used for stack overflows anyway.
                this.write_scalar(Scalar::from_int(1, dest.layout.size), dest)?;
            }
            "InitializeCriticalSection"
            | "EnterCriticalSection"
            | "LeaveCriticalSection"
            | "DeleteCriticalSection" => {
                // Nothing to do, not even a return value.
            }
            "GetModuleHandleW"
            | "GetProcAddress"
            | "TryEnterCriticalSection"
            | "GetConsoleScreenBufferInfo"
            | "SetConsoleTextAttribute" => {
                // Pretend these do not exist / nothing happened, by returning zero.
                this.write_null(dest)?;
            }
            "GetSystemInfo" => {
                let system_info = this.deref_operand(args[0])?;
                // Initialize with `0`.
                this.memory
                    .write_bytes(system_info.ptr, iter::repeat(0u8).take(system_info.layout.size.bytes() as usize))?;
                // Set number of processors.
                let dword_size = Size::from_bytes(4);
                let num_cpus = this.mplace_field(system_info, 6)?;
                this.write_scalar(
                    Scalar::from_int(NUM_CPUS, dword_size),
                    num_cpus.into(),
                )?;
            }

            "TlsAlloc" => {
                // This just creates a key; Windows does not natively support TLS destructors.

                // Create key and return it.
                let key = this.machine.tls.create_tls_key(None) as u128;

                // Figure out how large a TLS key actually is. This is `c::DWORD`.
                if dest.layout.size.bits() < 128
                    && key >= (1u128 << dest.layout.size.bits() as u128)
                {
                    throw_unsup!(OutOfTls);
                }
                this.write_scalar(Scalar::from_uint(key, dest.layout.size), dest)?;
            }
            "TlsGetValue" => {
                let key = this.read_scalar(args[0])?.to_u32()? as u128;
                let ptr = this.machine.tls.load_tls(key, tcx)?;
                this.write_scalar(ptr, dest)?;
            }
            "TlsSetValue" => {
                let key = this.read_scalar(args[0])?.to_u32()? as u128;
                let new_ptr = this.read_scalar(args[1])?.not_undef()?;
                this.machine.tls.store_tls(key, this.test_null(new_ptr)?)?;

                // Return success (`1`).
                this.write_scalar(Scalar::from_int(1, dest.layout.size), dest)?;
            }
            "GetStdHandle" => {
                let which = this.read_scalar(args[0])?.to_i32()?;
                // We just make this the identity function, so we know later in `WriteFile`
                // which one it is.
                this.write_scalar(Scalar::from_int(which, this.pointer_size()), dest)?;
            }
            "WriteFile" => {
                let handle = this.read_scalar(args[0])?.to_isize(this)?;
                let buf = this.read_scalar(args[1])?.not_undef()?;
                let n = this.read_scalar(args[2])?.to_u32()?;
                let written_place = this.deref_operand(args[3])?;
                // Spec says to always write `0` first.
                this.write_null(written_place.into())?;
                let written = if handle == -11 || handle == -12 {
                    // stdout/stderr
                    use std::io::{self, Write};

                    let buf_cont = this
                        .memory
                        .read_bytes(buf, Size::from_bytes(u64::from(n)))?;
                    let res = if handle == -11 {
                        io::stdout().write(buf_cont)
                    } else {
                        io::stderr().write(buf_cont)
                    };
                    res.ok().map(|n| n as u32)
                } else {
                    eprintln!("Miri: Ignored output to handle {}", handle);
                    // Pretend it all went well.
                    Some(n)
                };
                // If there was no error, write back how much was written.
                if let Some(n) = written {
                    this.write_scalar(Scalar::from_u32(n), written_place.into())?;
                }
                // Return whether this was a success.
                this.write_scalar(
                    Scalar::from_int(if written.is_some() { 1 } else { 0 }, dest.layout.size),
                    dest,
                )?;
            }
            "GetConsoleMode" => {
                // Everything is a pipe.
                this.write_null(dest)?;
            }
            "GetEnvironmentVariableW" => {
                // This is not the env var you are looking for.
                this.set_last_error(Scalar::from_u32(203))?; // ERROR_ENVVAR_NOT_FOUND
                this.write_null(dest)?;
            }
            "GetCommandLineW" => {
                this.write_scalar(Scalar::Ptr(this.machine.cmd_line.unwrap()), dest)?;
            }
            // The actual name of 'RtlGenRandom'
            "SystemFunction036" => {
                let ptr = this.read_scalar(args[0])?.not_undef()?;
                let len = this.read_scalar(args[1])?.to_u32()?;
                this.gen_random(ptr, len as usize)?;
                this.write_scalar(Scalar::from_bool(true), dest)?;
            }

            // We can't execute anything else.
            _ => throw_unsup_format!("can't call foreign function: {}", link_name),
        }

        this.goto_block(Some(ret))?;
        this.dump_place(*dest);
        Ok(())
    }

    /// Evaluates the scalar at the specified path. Returns Some(val)
    /// if the path could be resolved, and None otherwise
    fn eval_path_scalar(
        &mut self,
        path: &[&str],
    ) -> InterpResult<'tcx, Option<ScalarMaybeUndef<Tag>>> {
        let this = self.eval_context_mut();
        if let Ok(instance) = this.resolve_path(path) {
            let cid = GlobalId {
                instance,
                promoted: None,
            };
            let const_val = this.const_eval_raw(cid)?;
            let const_val = this.read_scalar(const_val.into())?;
            return Ok(Some(const_val));
        }
        return Ok(None);
    }
}

// Shims the linux 'getrandom()' syscall.
fn linux_getrandom<'tcx>(
    this: &mut MiriEvalContext<'_, 'tcx>,
    args: &[OpTy<'tcx, Tag>],
    dest: PlaceTy<'tcx, Tag>,
) -> InterpResult<'tcx> {
    let ptr = this.read_scalar(args[0])?.not_undef()?;
    let len = this.read_scalar(args[1])?.to_usize(this)?;

    // The only supported flags are GRND_RANDOM and GRND_NONBLOCK,
    // neither of which have any effect on our current PRNG.
    let _flags = this.read_scalar(args[2])?.to_i32()?;

    this.gen_random(ptr, len as usize)?;
    this.write_scalar(Scalar::from_uint(len, dest.layout.size), dest)?;
    Ok(())
}
