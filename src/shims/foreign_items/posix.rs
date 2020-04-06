mod linux;
mod macos;

use std::convert::TryFrom;

use log::trace;

use crate::*;
use rustc_middle::mir;
use rustc_target::abi::{Align, LayoutOf, Size};

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

        match link_name {
            // Environment related shims
            "getenv" => {
                let result = this.getenv(args[0])?;
                this.write_scalar(result, dest)?;
            }
            "unsetenv" => {
                let result = this.unsetenv(args[0])?;
                this.write_scalar(Scalar::from_i32(result), dest)?;
            }
            "setenv" => {
                let result = this.setenv(args[0], args[1])?;
                this.write_scalar(Scalar::from_i32(result), dest)?;
            }
            "getcwd" => {
                let result = this.getcwd(args[0], args[1])?;
                this.write_scalar(result, dest)?;
            }
            "chdir" => {
                let result = this.chdir(args[0])?;
                this.write_scalar(Scalar::from_i32(result), dest)?;
            }

            // File related shims
            "open" | "open64" => {
                let result = this.open(args[0], args[1])?;
                this.write_scalar(Scalar::from_i32(result), dest)?;
            }
            "fcntl" => {
                let result = this.fcntl(args[0], args[1], args.get(2).cloned())?;
                this.write_scalar(Scalar::from_i32(result), dest)?;
            }
            "read" => {
                let result = this.read(args[0], args[1], args[2])?;
                this.write_scalar(Scalar::from_machine_isize(result, this), dest)?;
            }
            "write" => {
                let fd = this.read_scalar(args[0])?.to_i32()?;
                let buf = this.read_scalar(args[1])?.not_undef()?;
                let n = this.read_scalar(args[2])?.to_machine_usize(this)?;
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
                        Ok(n) => i64::try_from(n).unwrap(),
                        Err(_) => -1,
                    }
                } else {
                    this.write(args[0], args[1], args[2])?
                };
                // Now, `result` is the value we return back to the program.
                this.write_scalar(Scalar::from_machine_isize(result, this), dest)?;
            }
            "unlink" => {
                let result = this.unlink(args[0])?;
                this.write_scalar(Scalar::from_i32(result), dest)?;
            }
            "symlink" => {
                let result = this.symlink(args[0], args[1])?;
                this.write_scalar(Scalar::from_i32(result), dest)?;
            }
            "rename" => {
                let result = this.rename(args[0], args[1])?;
                this.write_scalar(Scalar::from_i32(result), dest)?;
            }
            "mkdir" => {
                let result = this.mkdir(args[0], args[1])?;
                this.write_scalar(Scalar::from_i32(result), dest)?;
            }
            "rmdir" => {
                let result = this.rmdir(args[0])?;
                this.write_scalar(Scalar::from_i32(result), dest)?;
            }
            "closedir" => {
                let result = this.closedir(args[0])?;
                this.write_scalar(Scalar::from_i32(result), dest)?;
            }
            "lseek" | "lseek64" => {
                let result = this.lseek64(args[0], args[1], args[2])?;
                // "lseek" is only used on macOS which is 64bit-only, so `i64` always works.
                this.write_scalar(Scalar::from_i64(result), dest)?;
            }

            // Allocation
            "posix_memalign" => {
                let ret = this.deref_operand(args[0])?;
                let align = this.read_scalar(args[1])?.to_machine_usize(this)?;
                let size = this.read_scalar(args[2])?.to_machine_usize(this)?;
                // Align must be power of 2, and also at least ptr-sized (POSIX rules).
                if !align.is_power_of_two() {
                    throw_ub_format!("posix_memalign: alignment must be a power of two, but is {}", align);
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

            // Dynamic symbol loading
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

            // Querying system information
            "sysconf" => {
                let name = this.read_scalar(args[0])?.to_i32()?;

                let sysconfs = &[
                    ("_SC_PAGESIZE", Scalar::from_int(PAGE_SIZE, this.pointer_size())),
                    ("_SC_NPROCESSORS_ONLN", Scalar::from_int(NUM_CPUS, this.pointer_size())),
                ];
                let mut result = None;
                for &(sysconf_name, value) in sysconfs {
                    let sysconf_name = this.eval_libc_i32(sysconf_name)?;
                    if sysconf_name == name {
                        result = Some(value);
                        break;
                    }
                }
                if let Some(result) = result {
                    this.write_scalar(result, dest)?;
                } else {
                    throw_unsup_format!("unimplemented sysconf name: {}", name)
                }
            }

            // Thread-local storage
            "pthread_key_create" => {
                let key_place = this.deref_operand(args[0])?;

                // Extract the function type out of the signature (that seems easier than constructing it ourselves).
                let dtor = match this.test_null(this.read_scalar(args[1])?.not_undef()?)? {
                    Some(dtor_ptr) => Some(this.memory.get_fn(dtor_ptr)?.as_instance()?),
                    None => None,
                };

                // Figure out how large a pthread TLS key actually is.
                // To this end, deref the argument type. This is `libc::pthread_key_t`.
                let key_type = args[0].layout.ty
                    .builtin_deref(true)
                    .ok_or_else(|| err_ub_format!(
                        "wrong signature used for `pthread_key_create`: first argument must be a raw pointer."
                    ))?
                    .ty;
                let key_layout = this.layout_of(key_type)?;

                // Create key and write it into the memory where `key_ptr` wants it.
                let key = this.machine.tls.create_tls_key(dtor, key_layout.size)?;
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
                let ptr = this.machine.tls.load_tls(key, this)?;
                this.write_scalar(ptr, dest)?;
            }
            "pthread_setspecific" => {
                let key = this.force_bits(this.read_scalar(args[0])?.not_undef()?, args[0].layout.size)?;
                let new_ptr = this.read_scalar(args[1])?.not_undef()?;
                this.machine.tls.store_tls(key, this.test_null(new_ptr)?)?;

                // Return success (`0`).
                this.write_null(dest)?;
            }

            // Synchronization primitives
            "pthread_mutexattr_init" => {
                let result = this.pthread_mutexattr_init(args[0])?;
                this.write_scalar(Scalar::from_i32(result), dest)?;
            }
            "pthread_mutexattr_settype" => {
                let result = this.pthread_mutexattr_settype(args[0], args[1])?;
                this.write_scalar(Scalar::from_i32(result), dest)?;
            }
            "pthread_mutexattr_destroy" => {
                let result = this.pthread_mutexattr_destroy(args[0])?;
                this.write_scalar(Scalar::from_i32(result), dest)?;
            }
            "pthread_mutex_init" => {
                let result = this.pthread_mutex_init(args[0], args[1])?;
                this.write_scalar(Scalar::from_i32(result), dest)?;
            }
            "pthread_mutex_lock" => {
                let result = this.pthread_mutex_lock(args[0])?;
                this.write_scalar(Scalar::from_i32(result), dest)?;
            }
            "pthread_mutex_trylock" => {
                let result = this.pthread_mutex_trylock(args[0])?;
                this.write_scalar(Scalar::from_i32(result), dest)?;
            }
            "pthread_mutex_unlock" => {
                let result = this.pthread_mutex_unlock(args[0])?;
                this.write_scalar(Scalar::from_i32(result), dest)?;
            }
            "pthread_mutex_destroy" => {
                let result = this.pthread_mutex_destroy(args[0])?;
                this.write_scalar(Scalar::from_i32(result), dest)?;
            }
            "pthread_rwlock_rdlock" => {
                let result = this.pthread_rwlock_rdlock(args[0])?;
                this.write_scalar(Scalar::from_i32(result), dest)?;
            }
            "pthread_rwlock_tryrdlock" => {
                let result = this.pthread_rwlock_tryrdlock(args[0])?;
                this.write_scalar(Scalar::from_i32(result), dest)?;
            }
            "pthread_rwlock_wrlock" => {
                let result = this.pthread_rwlock_wrlock(args[0])?;
                this.write_scalar(Scalar::from_i32(result), dest)?;
            }
            "pthread_rwlock_trywrlock" => {
                let result = this.pthread_rwlock_trywrlock(args[0])?;
                this.write_scalar(Scalar::from_i32(result), dest)?;
            }
            "pthread_rwlock_unlock" => {
                let result = this.pthread_rwlock_unlock(args[0])?;
                this.write_scalar(Scalar::from_i32(result), dest)?;
            }
            "pthread_rwlock_destroy" => {
                let result = this.pthread_rwlock_destroy(args[0])?;
                this.write_scalar(Scalar::from_i32(result), dest)?;
            }

            // Better error for attempts to create a thread
            "pthread_create" => {
                throw_unsup_format!("Miri does not support threading");
            }

            // Miscellaneous
            "isatty" => {
                let _fd = this.read_scalar(args[0])?.to_i32()?;
                // "returns 1 if fd is an open file descriptor referring to a terminal; otherwise 0 is returned, and errno is set to indicate the error"
                // FIXME: we just say nothing is a terminal.
                let enotty = this.eval_libc("ENOTTY")?;
                this.set_last_error(enotty)?;
                this.write_null(dest)?;
            }
            "pthread_atfork" => {
                let _prepare = this.read_scalar(args[0])?.not_undef()?;
                let _parent = this.read_scalar(args[1])?.not_undef()?;
                let _child = this.read_scalar(args[1])?.not_undef()?;
                // We do not support forking, so there is nothing to do here.
                this.write_null(dest)?;
            }

            // Incomplete shims that we "stub out" just to get pre-main initialization code to work.
            // These shims are enabled only when the caller is in the standard library.
            | "pthread_attr_init"
            | "pthread_attr_destroy"
            | "pthread_self"
            | "pthread_attr_setstacksize"
            | "pthread_condattr_init"
            | "pthread_condattr_setclock"
            | "pthread_cond_init"
            | "pthread_condattr_destroy"
            | "pthread_cond_destroy" if this.frame().instance.to_string().starts_with("std::sys::unix::")
            => {
                this.write_null(dest)?;
            }

            | "signal"
            | "sigaction"
            | "sigaltstack"
            | "mprotect" if this.frame().instance.to_string().starts_with("std::sys::unix::")
            => {
                this.write_null(dest)?;
            }

            // Platform-specific shims
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
