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
        let ecx = self.eval_context_mut();
        match link_name.as_str() {
            // Threading
            "pthread_setname_np" => {
                let [thread, name] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(libc::pthread_t, *_) -> i32),
                    (link_name, abi, args),
                )?;
                let max_len = u64::MAX; // FreeBSD does not seem to have a limit.
                let res = match ecx.pthread_setname_np(
                    Some(thread),
                    ecx.read_pointer(name)?,
                    max_len,
                    /* truncate */ false,
                )? {
                    ThreadNameResult::Ok => Scalar::from_u32(0),
                    ThreadNameResult::NameTooLong => unreachable!(),
                };
                ecx.write_scalar(res, dest)?;
            }
            "pthread_getname_np" => {
                let [thread, name, len] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(libc::pthread_t, *_, usize) -> i32),
                    (link_name, abi, args),
                )?;
                // FreeBSD's pthread_getname_np uses strlcpy, which truncates the resulting value,
                // but always adds a null terminator (except for zero-sized buffers).
                // https://github.com/freebsd/freebsd-src/blob/c2d93a803acef634bd0eede6673aeea59e90c277/lib/libthr/thread/thr_info.c#L119-L144
                let res = match ecx.pthread_getname_np(
                    Some(thread),
                    ecx.read_pointer(name)?,
                    ecx.read_scalar(len)?,
                    /* truncate */ true,
                )? {
                    ThreadNameResult::Ok => Scalar::from_u32(0),
                    // `NameTooLong` is possible when the buffer is zero sized,
                    ThreadNameResult::NameTooLong => Scalar::from_u32(0),
                };
                ecx.write_scalar(res, dest)?;
            }
            "pthread_getthreadid_np" => {
                let [] =
                    &ecx.check_shim_sig(shim_sig!(extern "C" fn() -> i32), (link_name, abi, args))?;
                let result = ecx.unix_gettid(link_name.as_str())?;
                ecx.write_scalar(result, dest)?;
            }

            "cpuset_getaffinity" => {
                // The "same" kind of api as `sched_getaffinity` but more fine grained control for FreeBSD specifically.
                let [level, which, id, set_size, mask] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(i32, i32, i64, usize, *_) -> i32),
                    (link_name, abi, args),
                )?;

                let level = ecx.read_scalar(level)?.to_i32()?;
                let which = ecx.read_scalar(which)?.to_i32()?;
                let id = ecx.read_scalar(id)?.to_i64()?;
                let set_size = ecx.read_target_usize(set_size)?; // measured in bytes
                let mask = ecx.read_pointer(mask)?;

                if ecx.machine.thread_cpu_affinity.is_none() {
                    throw_unsup_format!(
                        "`cpuset_getaffinity` is not supported on #![no_core] programs"
                    )
                }

                let _level_root = ecx.eval_libc_i32("CPU_LEVEL_ROOT");
                let _level_cpuset = ecx.eval_libc_i32("CPU_LEVEL_CPUSET");
                let level_which = ecx.eval_libc_i32("CPU_LEVEL_WHICH");

                let _which_tid = ecx.eval_libc_i32("CPU_WHICH_TID");
                let which_pid = ecx.eval_libc_i32("CPU_WHICH_PID");
                let _which_jail = ecx.eval_libc_i32("CPU_WHICH_JAIL");
                let _which_cpuset = ecx.eval_libc_i32("CPU_WHICH_CPUSET");
                let _which_irq = ecx.eval_libc_i32("CPU_WHICH_IRQ");

                // For sched_getaffinity, the current process is identified by -1.
                // TODO: Use gettid? I'm (LorrensP-2158466) not that familiar with this api .
                let id = match id {
                    -1 => ecx.active_thread(),
                    _ =>
                        throw_unsup_format!(
                            "`cpuset_getaffinity` is only supported with a pid of -1 (indicating the current thread)"
                        ),
                };

                if ecx.ptr_is_null(mask)? {
                    ecx.set_errno_and_return_neg1(LibcError("EFAULT"), dest)?;
                }
                // We only support CPU_LEVEL_WHICH and CPU_WHICH_PID for now.
                // This is the bare minimum to make the tests pass.
                else if level != level_which || which != which_pid {
                    throw_unsup_format!(
                        "`cpuset_getaffinity` is only supported with `level` set to CPU_LEVEL_WHICH and `which` set to CPU_WHICH_PID."
                    );
                } else if let Some(cpuset) =
                    ecx.machine.thread_cpu_affinity.as_ref().unwrap().get(&id)
                {
                    // `cpusetsize` must be large enough to contain the entire CPU mask.
                    // FreeBSD only uses `cpusetsize` to verify that it's sufficient for the kernel's CPU mask.
                    // If it's too small, the syscall returns ERANGE.
                    // If it's large enough, copying the kernel mask to user space is safe, regardless of the actual size.
                    // See https://github.com/freebsd/freebsd-src/blob/909aa6781340f8c0b4ae01c6366bf1556ee2d1be/sys/kern/kern_cpuset.c#L1985
                    if set_size < u64::from(ecx.machine.num_cpus).div_ceil(8) {
                        ecx.set_errno_and_return_neg1(LibcError("ERANGE"), dest)?;
                    } else {
                        let cpuset = cpuset.clone();
                        let byte_count =
                            Ord::min(cpuset.as_slice().len(), set_size.try_into().unwrap());
                        ecx.write_bytes_ptr(mask, cpuset.as_slice()[..byte_count].iter().copied())?;
                        ecx.write_int(0, dest)?;
                    }
                } else {
                    // `id` is always that of the active thread, so this is currently unreachable.
                    unreachable!();
                }
            }

            // Synchronization primitives
            "_umtx_op" => {
                let [obj, op, val, uaddr, uaddr2] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(*_, i32, usize, *_, *_) -> i32),
                    (link_name, abi, args),
                )?;
                ecx._umtx_op(obj, op, val, uaddr, uaddr2, dest)?;
            }

            // Miscellaneous
            "__error" => {
                let [] =
                    &ecx.check_shim_sig(shim_sig!(extern "C" fn() -> *_), (link_name, abi, args))?;
                let errno_place = ecx.last_error_place()?;
                ecx.write_scalar(errno_place.to_ref(ecx).to_scalar(), dest)?;
            }
            "__xuname" => {
                // FreeBSD uses __xuname under the hood to implement uname, see:
                // https://github.com/freebsd/freebsd-src/blob/3542d60fb8042474f66fbf2d779ed8c5a80d0f78/sys/sys/utsname.h#L64
                // https://github.com/freebsd/freebsd-src/blob/3542d60fb8042474f66fbf2d779ed8c5a80d0f78/lib/libc/gen/uname.c#L44
                let [size, uname] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(i32, *libc::utsname) -> i32),
                    (link_name, abi, args),
                )?;
                let result = ecx.uname(uname, Some(size))?;
                ecx.write_scalar(result, dest)?;
            }

            // Incomplete shims that we "stub out" just to get pre-main initialization code to work.
            // These shims are enabled only when the caller is in the standard library.
            "pthread_attr_get_np" if ecx.frame_in_std() => {
                let [_thread, attr] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(libc::pthread_t, *libc::pthread_attr_t) -> i32),
                    (link_name, abi, args),
                )?;
                // The caller will `pthread_attr_destroy` this, so it needs to be initialized.
                ecx.pthread_attr_init(attr, dest)?;
            }

            _ => return interp_ok(EmulateItemResult::NotSupported),
        }
        interp_ok(EmulateItemResult::NeedsReturn)
    }
}
