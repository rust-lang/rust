use rustc_abi::CanonAbi;
use rustc_middle::ty::Ty;
use rustc_span::Symbol;
use rustc_target::callconv::FnAbi;

use super::sync::EvalContextExt as _;
use crate::shims::unix::*;
use crate::*;

pub fn is_dyn_sym(_name: &str) -> bool {
    false
}

impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
pub trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    fn emulate_foreign_item_inner(
        &mut self,
        link_name: Symbol,
        abi: &FnAbi<'tcx, Ty<'tcx>>,
        args: &[OpTy<'tcx>],
        dest: &MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx, EmulateItemResult> {
        let this = self.eval_context_mut();
        match link_name.as_str() {
            // Threading
            "pthread_setname_np" => {
                let [thread, name] = this.check_shim(abi, CanonAbi::C, link_name, args)?;
                let max_len = u64::MAX; // FreeBSD does not seem to have a limit.
                let res = match this.pthread_setname_np(
                    this.read_scalar(thread)?,
                    this.read_scalar(name)?,
                    max_len,
                    /* truncate */ false,
                )? {
                    ThreadNameResult::Ok => Scalar::from_u32(0),
                    ThreadNameResult::NameTooLong => unreachable!(),
                    ThreadNameResult::ThreadNotFound => this.eval_libc("ESRCH"),
                };
                this.write_scalar(res, dest)?;
            }
            "pthread_getname_np" => {
                let [thread, name, len] = this.check_shim(abi, CanonAbi::C, link_name, args)?;
                // FreeBSD's pthread_getname_np uses strlcpy, which truncates the resulting value,
                // but always adds a null terminator (except for zero-sized buffers).
                // https://github.com/freebsd/freebsd-src/blob/c2d93a803acef634bd0eede6673aeea59e90c277/lib/libthr/thread/thr_info.c#L119-L144
                let res = match this.pthread_getname_np(
                    this.read_scalar(thread)?,
                    this.read_scalar(name)?,
                    this.read_scalar(len)?,
                    /* truncate */ true,
                )? {
                    ThreadNameResult::Ok => Scalar::from_u32(0),
                    // `NameTooLong` is possible when the buffer is zero sized,
                    ThreadNameResult::NameTooLong => Scalar::from_u32(0),
                    ThreadNameResult::ThreadNotFound => this.eval_libc("ESRCH"),
                };
                this.write_scalar(res, dest)?;
            }

            "cpuset_getaffinity" => {
                // The "same" kind of api as `sched_getaffinity` but more fine grained control for FreeBSD specifically.
                let [level, which, id, set_size, mask] =
                    this.check_shim(abi, CanonAbi::C, link_name, args)?;

                let level = this.read_scalar(level)?.to_i32()?;
                let which = this.read_scalar(which)?.to_i32()?;
                let id = this.read_scalar(id)?.to_i64()?;
                let set_size = this.read_target_usize(set_size)?; // measured in bytes
                let mask = this.read_pointer(mask)?;

                let _level_root = this.eval_libc_i32("CPU_LEVEL_ROOT");
                let _level_cpuset = this.eval_libc_i32("CPU_LEVEL_CPUSET");
                let level_which = this.eval_libc_i32("CPU_LEVEL_WHICH");

                let _which_tid = this.eval_libc_i32("CPU_WHICH_TID");
                let which_pid = this.eval_libc_i32("CPU_WHICH_PID");
                let _which_jail = this.eval_libc_i32("CPU_WHICH_JAIL");
                let _which_cpuset = this.eval_libc_i32("CPU_WHICH_CPUSET");
                let _which_irq = this.eval_libc_i32("CPU_WHICH_IRQ");

                // For sched_getaffinity, the current process is identified by -1.
                // TODO: Use gettid? I'm (LorrensP-2158466) not that familiar with this api .
                let id = match id {
                    -1 => this.active_thread(),
                    _ =>
                        throw_unsup_format!(
                            "`cpuset_getaffinity` is only supported with a pid of -1 (indicating the current thread)"
                        ),
                };

                if this.ptr_is_null(mask)? {
                    this.set_last_error_and_return(LibcError("EFAULT"), dest)?;
                }
                // We only support CPU_LEVEL_WHICH and CPU_WHICH_PID for now.
                // This is the bare minimum to make the tests pass.
                else if level != level_which || which != which_pid {
                    throw_unsup_format!(
                        "`cpuset_getaffinity` is only supported with `level` set to CPU_LEVEL_WHICH and `which` set to CPU_WHICH_PID."
                    );
                } else if let Some(cpuset) = this.machine.thread_cpu_affinity.get(&id) {
                    // `cpusetsize` must be large enough to contain the entire CPU mask.
                    // FreeBSD only uses `cpusetsize` to verify that it's sufficient for the kernel's CPU mask.
                    // If it's too small, the syscall returns ERANGE.
                    // If it's large enough, copying the kernel mask to user space is safe, regardless of the actual size.
                    // See https://github.com/freebsd/freebsd-src/blob/909aa6781340f8c0b4ae01c6366bf1556ee2d1be/sys/kern/kern_cpuset.c#L1985
                    if set_size < u64::from(this.machine.num_cpus).div_ceil(8) {
                        this.set_last_error_and_return(LibcError("ERANGE"), dest)?;
                    } else {
                        let cpuset = cpuset.clone();
                        let byte_count =
                            Ord::min(cpuset.as_slice().len(), set_size.try_into().unwrap());
                        this.write_bytes_ptr(
                            mask,
                            cpuset.as_slice()[..byte_count].iter().copied(),
                        )?;
                        this.write_null(dest)?;
                    }
                } else {
                    // `id` is always that of the active thread, so this is currently unreachable.
                    unreachable!();
                }
            }

            // Synchronization primitives
            "_umtx_op" => {
                let [obj, op, val, uaddr, uaddr2] =
                    this.check_shim(abi, CanonAbi::C, link_name, args)?;
                this._umtx_op(obj, op, val, uaddr, uaddr2, dest)?;
            }

            // File related shims
            // For those, we both intercept `func` and `call@FBSD_1.0` symbols cases
            // since freebsd 12 the former form can be expected.
            "stat" | "stat@FBSD_1.0" => {
                let [path, buf] = this.check_shim(abi, CanonAbi::C, link_name, args)?;
                let result = this.macos_fbsd_solarish_stat(path, buf)?;
                this.write_scalar(result, dest)?;
            }
            "lstat" | "lstat@FBSD_1.0" => {
                let [path, buf] = this.check_shim(abi, CanonAbi::C, link_name, args)?;
                let result = this.macos_fbsd_solarish_lstat(path, buf)?;
                this.write_scalar(result, dest)?;
            }
            "fstat" | "fstat@FBSD_1.0" => {
                let [fd, buf] = this.check_shim(abi, CanonAbi::C, link_name, args)?;
                let result = this.macos_fbsd_solarish_fstat(fd, buf)?;
                this.write_scalar(result, dest)?;
            }
            "readdir_r" | "readdir_r@FBSD_1.0" => {
                let [dirp, entry, result] = this.check_shim(abi, CanonAbi::C, link_name, args)?;
                let result = this.macos_fbsd_readdir_r(dirp, entry, result)?;
                this.write_scalar(result, dest)?;
            }

            // Miscellaneous
            "__error" => {
                let [] = this.check_shim(abi, CanonAbi::C, link_name, args)?;
                let errno_place = this.last_error_place()?;
                this.write_scalar(errno_place.to_ref(this).to_scalar(), dest)?;
            }

            // Incomplete shims that we "stub out" just to get pre-main initialization code to work.
            // These shims are enabled only when the caller is in the standard library.
            "pthread_attr_get_np" if this.frame_in_std() => {
                let [_thread, _attr] = this.check_shim(abi, CanonAbi::C, link_name, args)?;
                this.write_null(dest)?;
            }

            _ => return interp_ok(EmulateItemResult::NotSupported),
        }
        interp_ok(EmulateItemResult::NeedsReturn)
    }
}
