mod linux;
mod macos;

use crate::*;
use rustc::mir;
use rustc::ty::layout::{Align, LayoutOf, Size};

impl<'mir, 'tcx> EvalContextExt<'mir, 'tcx> for crate::MiriEvalContext<'mir, 'tcx> {}
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriEvalContextExt<'mir, 'tcx> {
    fn emulate_foreign_item_by_name(
        &mut self,
        link_name: &str,
        args: &[OpTy<'tcx, Tag>],
        dest: PlaceTy<'tcx, Tag>,
        ret: mir::BasicBlock,
    ) -> InterpResult<'tcx, bool> {
        let this = self.eval_context_mut();
        let tcx = &{ this.tcx.tcx };

        match link_name {
            // Environment related shims
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

            // File related shims
            "open" | "open64" => {
                let result = this.open(args[0], args[1])?;
                this.write_scalar(Scalar::from_int(result, dest.layout.size), dest)?;
            }

            "fcntl" => {
                let result = this.fcntl(args[0], args[1], args.get(2).cloned())?;
                this.write_scalar(Scalar::from_int(result, dest.layout.size), dest)?;
            }

            "read" => {
                let result = this.read(args[0], args[1], args[2])?;
                this.write_scalar(Scalar::from_int(result, dest.layout.size), dest)?;
            }

            "write" => {
                let fd = this.read_scalar(args[0])?.to_i32()?;
                let buf = this.read_scalar(args[1])?.not_undef()?;
                let n = this.read_scalar(args[2])?.to_machine_usize(tcx)?;
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

            "symlink" => {
                let result = this.symlink(args[0], args[1])?;
                this.write_scalar(Scalar::from_int(result, dest.layout.size), dest)?;
            }

            "rename" => {
                let result = this.rename(args[0], args[1])?;
                this.write_scalar(Scalar::from_int(result, dest.layout.size), dest)?;
            }

            "mkdir" => {
                let result = this.mkdir(args[0], args[1])?;
                this.write_scalar(Scalar::from_int(result, dest.layout.size), dest)?;
            }

            "rmdir" => {
                let result = this.rmdir(args[0])?;
                this.write_scalar(Scalar::from_int(result, dest.layout.size), dest)?;
            }

            "closedir" => {
                let result = this.closedir(args[0])?;
                this.write_scalar(Scalar::from_int(result, dest.layout.size), dest)?;
            }

            "lseek" | "lseek64" => {
                let result = this.lseek64(args[0], args[1], args[2])?;
                this.write_scalar(Scalar::from_int(result, dest.layout.size), dest)?;
            }

            // Other shims
            "posix_memalign" => {
                let ret = this.deref_operand(args[0])?;
                let align = this.read_scalar(args[1])?.to_machine_usize(this)?;
                let size = this.read_scalar(args[2])?.to_machine_usize(this)?;
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
                    this.write_scalar(ptr, ret.into())?;
                }
                this.write_null(dest)?;
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
                let key = this.force_bits(this.read_scalar(args[0])?.not_undef()?, args[0].layout.size)?;
                this.machine.tls.delete_tls_key(key)?;
                // Return success (0)
                this.write_null(dest)?;
            }
            "pthread_getspecific" => {
                let key = this.force_bits(this.read_scalar(args[0])?.not_undef()?, args[0].layout.size)?;
                let ptr = this.machine.tls.load_tls(key, tcx)?;
                this.write_scalar(ptr, dest)?;
            }
            "pthread_setspecific" => {
                let key = this.force_bits(this.read_scalar(args[0])?.not_undef()?, args[0].layout.size)?;
                let new_ptr = this.read_scalar(args[1])?.not_undef()?;
                this.machine.tls.store_tls(key, this.test_null(new_ptr)?)?;

                // Return success (`0`).
                this.write_null(dest)?;
            }

            // Stack size/address stuff.
            | "pthread_attr_init"
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

            // We don't support threading.
            "pthread_create" => {
                throw_unsup_format!("Miri does not support threading");
            }

            // Stub out calls for condvar, mutex and rwlock, to just return `0`.
            | "pthread_mutexattr_init"
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
            | "pthread_cond_destroy"
            => {
                this.write_null(dest)?;
            }

            // We don't support fork so we don't have to do anything for atfork.
            "pthread_atfork" => {
                this.write_null(dest)?;
            }

            // Some things needed for `sys::thread` initialization to go through.
            | "signal"
            | "sigaction"
            | "sigaltstack"
            => {
                this.write_scalar(Scalar::from_int(0, dest.layout.size), dest)?;
            }

            "sysconf" => {
                let name = this.read_scalar(args[0])?.to_i32()?;

                trace!("sysconf() called with name {}", name);
                // TODO: Cache the sysconf integers via Miri's global cache.
                let paths = &[
                    (&["libc", "_SC_PAGESIZE"], Scalar::from_int(PAGE_SIZE, dest.layout.size)),
                    (&["libc", "_SC_GETPW_R_SIZE_MAX"], Scalar::from_int(-1, dest.layout.size)),
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

            "isatty" => {
                this.write_null(dest)?;
            }

            "posix_fadvise" => {
                // fadvise is only informational, we can ignore it.
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

            _ => {
                match this.tcx.sess.target.target.target_os.as_str() {
                    "linux" => return linux::EvalContextExt::emulate_foreign_item_by_name(this, link_name, args, dest, ret),
                    "macos" => return macos::EvalContextExt::emulate_foreign_item_by_name(this, link_name, args, dest, ret),
                    _ => unreachable!(),
                }
            }
        };

        Ok(true)
    }
}
