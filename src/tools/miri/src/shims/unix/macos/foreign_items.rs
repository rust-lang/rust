use rustc_middle::ty::Ty;
use rustc_span::Symbol;
use rustc_target::callconv::FnAbi;

use super::sync::{EvalContextExt as _, MacOsFutexTimeout};
use crate::shims::unix::*;
use crate::*;

pub fn is_dyn_sym(name: &str) -> bool {
    match name {
        // These only became available with macOS 11.0, so std looks them up dynamically.
        "os_sync_wait_on_address"
        | "os_sync_wait_on_address_with_deadline"
        | "os_sync_wait_on_address_with_timeout"
        | "os_sync_wake_by_address_any"
        | "os_sync_wake_by_address_all" => true,
        _ => false,
    }
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

        // See `fn emulate_foreign_item_inner` in `shims/foreign_items.rs` for the general pattern.

        match link_name.as_str() {
            // errno
            "__error" => {
                let [] =
                    &ecx.check_shim_sig(shim_sig!(extern "C" fn() -> *_), (link_name, abi, args))?;
                let errno_place = ecx.last_error_place()?;
                ecx.write_scalar(errno_place.to_ref(ecx).to_scalar(), dest)?;
            }

            // File related shims
            "close$NOCANCEL" => {
                let [fd] = &ecx
                    .check_shim_sig(shim_sig!(extern "C" fn(i32) -> i32), (link_name, abi, args))?;
                let fd = ecx.read_scalar(fd)?.to_i32()?;
                let result = ecx.close(fd)?;
                ecx.write_scalar(result, dest)?;
            }
            "stat$INODE64" => {
                let [path, buf] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(*_, *libc::stat) -> i32),
                    (link_name, abi, args),
                )?;
                let result = ecx.stat(path, buf)?;
                ecx.write_scalar(result, dest)?;
            }
            "lstat$INODE64" => {
                let [path, buf] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(*_, *libc::stat) -> i32),
                    (link_name, abi, args),
                )?;
                let result = ecx.lstat(path, buf)?;
                ecx.write_scalar(result, dest)?;
            }
            "fstat$INODE64" => {
                let [fd, buf] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(i32, *libc::stat) -> i32),
                    (link_name, abi, args),
                )?;
                let result = ecx.fstat(fd, buf)?;
                ecx.write_scalar(result, dest)?;
            }
            "fstatat$INODE64" => {
                let [dirfd, path, buf, flags] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(i32, *_, *libc::stat, i32) -> i32),
                    (link_name, abi, args),
                )?;
                let result = ecx.fstatat(dirfd, path, buf, flags)?;
                ecx.write_scalar(result, dest)?;
            }
            "opendir$INODE64" => {
                let [name] = &ecx
                    .check_shim_sig(shim_sig!(extern "C" fn(*_) -> *_), (link_name, abi, args))?;
                let result = ecx.opendir(name)?;
                ecx.write_scalar(result, dest)?;
            }
            "readdir$INODE64" => {
                let [dirp] = &ecx
                    .check_shim_sig(shim_sig!(extern "C" fn(*_) -> *_), (link_name, abi, args))?;
                ecx.readdir(dirp, dest, "dirent")?;
            }
            "realpath$DARWIN_EXTSN" => {
                let [path, resolved_path] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(*_, *_) -> *_),
                    (link_name, abi, args),
                )?;
                let result = ecx.realpath(path, resolved_path)?;
                ecx.write_scalar(result, dest)?;
            }

            // Environment related shims
            "_NSGetEnviron" => {
                let [] =
                    &ecx.check_shim_sig(shim_sig!(extern "C" fn() -> *_), (link_name, abi, args))?;
                let environ = ecx.machine.env_vars.unix().environ();
                ecx.write_pointer(environ, dest)?;
            }

            // Random data generation
            "CCRandomGenerateBytes" => {
                let [bytes, count] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(*_, usize) -> i32),
                    (link_name, abi, args),
                )?;
                let bytes = ecx.read_pointer(bytes)?;
                let count = ecx.read_target_usize(count)?;
                let success = ecx.eval_libc_i32("kCCSuccess");
                ecx.gen_random(bytes, count)?;
                ecx.write_int(success, dest)?;
            }

            // Time related shims
            "mach_absolute_time" => {
                let [] =
                    &ecx.check_shim_sig(shim_sig!(extern "C" fn() -> u64), (link_name, abi, args))?;
                let result = ecx.mach_absolute_time()?;
                ecx.write_scalar(result, dest)?;
            }

            "mach_timebase_info" => {
                let [info] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(*libc::mach_timebase_info) -> i32),
                    (link_name, abi, args),
                )?;
                let result = ecx.mach_timebase_info(info)?;
                ecx.write_scalar(result, dest)?;
            }

            "mach_wait_until" => {
                // FIXME: This does not have a direct test (#3179).
                let [deadline] = &ecx
                    .check_shim_sig(shim_sig!(extern "C" fn(u64) -> i32), (link_name, abi, args))?;
                let result = ecx.mach_wait_until(deadline)?;
                ecx.write_scalar(result, dest)?;
            }

            // Access to command-line arguments
            "_NSGetArgc" => {
                let [] =
                    &ecx.check_shim_sig(shim_sig!(extern "C" fn() -> *_), (link_name, abi, args))?;
                ecx.write_pointer(ecx.machine.argc.expect("machine must be initialized"), dest)?;
            }
            "_NSGetArgv" => {
                let [] =
                    &ecx.check_shim_sig(shim_sig!(extern "C" fn() -> *_), (link_name, abi, args))?;
                ecx.write_pointer(ecx.machine.argv.expect("machine must be initialized"), dest)?;
            }
            "_NSGetExecutablePath" => {
                // FIXME: This does not have a direct test (#3179).
                let [buf, bufsize] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(*_, *u32) -> i32),
                    (link_name, abi, args),
                )?;
                ecx.check_no_isolation("`_NSGetExecutablePath`")?;

                let buf_ptr = ecx.read_pointer(buf)?;
                let bufsize = ecx.deref_pointer(bufsize)?;

                let Some(path) = ecx.machine.current_exe.clone() else {
                    throw_unsup_format!(
                        "_NSGetExecutablePath only works when Miri is run on a file"
                    )
                };
                let (written, size_needed) = ecx.write_path_to_c_str(
                    &path,
                    buf_ptr,
                    ecx.read_scalar(&bufsize)?.to_u32()?.into(),
                )?;

                if written {
                    ecx.write_int(0, dest)?;
                } else {
                    ecx.write_scalar(Scalar::from_u32(size_needed.try_into().unwrap()), &bufsize)?;
                    ecx.write_int(-1, dest)?;
                }
            }

            // Thread-local storage
            "_tlv_atexit" => {
                let [dtor, data] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(fn(..) -> _, *_) -> ()),
                    (link_name, abi, args),
                )?;
                let dtor = ecx.read_pointer(dtor)?;
                let dtor = ecx.get_ptr_fn(dtor)?.as_instance()?;
                let data = ecx.read_scalar(data)?;
                ecx.machine.tls.add_thread_atexit(
                    ecx.active_thread(),
                    dtor,
                    data,
                    ecx.machine.current_user_relevant_span(),
                )?;
            }

            // Querying system information
            "pthread_get_stackaddr_np" => {
                let [thread] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(libc::pthread_t) -> *_),
                    (link_name, abi, args),
                )?;
                ecx.read_target_usize(thread)?;
                ecx.write_int(ecx.machine.stack_addr, dest)?;
            }
            "pthread_get_stacksize_np" => {
                let [thread] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(libc::pthread_t) -> usize),
                    (link_name, abi, args),
                )?;
                ecx.read_target_usize(thread)?;
                ecx.write_int(ecx.machine.stack_size, dest)?;
            }

            // Threading
            "pthread_setname_np" => {
                let [name] = &ecx
                    .check_shim_sig(shim_sig!(extern "C" fn(*_) -> i32), (link_name, abi, args))?;

                // The real implementation has logic in two places:
                // * in userland at https://github.com/apple-oss-distributions/libpthread/blob/c032e0b076700a0a47db75528a282b8d3a06531a/src/pthread.c#L1178-L1200,
                // * in kernel at https://github.com/apple-oss-distributions/xnu/blob/8d741a5de7ff4191bf97d57b9f54c2f6d4a15585/bsd/kern/proc_info.c#L3218-L3227.
                //
                // The function in libc calls the kernel to validate
                // the security policies and the input. If all of the requirements
                // are met, then the name is set and 0 is returned. Otherwise, if
                // the specified name is lomnger than MAXTHREADNAMESIZE, then
                // ENAMETOOLONG is returned.
                let res = match ecx.pthread_setname_np(
                    None,
                    ecx.read_pointer(name)?,
                    ecx.eval_libc("MAXTHREADNAMESIZE").to_target_usize(ecx)?,
                    /* truncate */ false,
                )? {
                    ThreadNameResult::Ok => Scalar::from_i32(0),
                    ThreadNameResult::NameTooLong => {
                        // Testing on native systems indicates that the error is returned via errno.
                        ecx.set_errno_and_return_neg1_i32(LibcError("ENAMETOOLONG"))?
                    }
                };
                // Contrary to the manpage, `pthread_setname_np` on macOS still
                // returns an integer indicating success.
                ecx.write_scalar(res, dest)?;
            }
            "pthread_getname_np" => {
                let [thread, name, len] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(libc::pthread_t, *_, usize) -> i32),
                    (link_name, abi, args),
                )?;

                // The function's behavior isn't portable between platforms.
                // In case of macOS, a truncated name (due to a too small buffer)
                // does not lead to an error.
                //
                // For details, see the implementation at
                // https://github.com/apple-oss-distributions/libpthread/blob/c032e0b076700a0a47db75528a282b8d3a06531a/src/pthread.c#L1160-L1175.
                // The key part is the strlcpy, which truncates the resulting value,
                // but always null terminates (except for zero sized buffers).
                let res = match ecx.pthread_getname_np(
                    Some(thread),
                    ecx.read_pointer(name)?,
                    ecx.read_scalar(len)?,
                    /* truncate */ true,
                )? {
                    ThreadNameResult::Ok => Scalar::from_i32(0),
                    // `NameTooLong` is possible when the buffer is zero sized,
                    ThreadNameResult::NameTooLong => Scalar::from_i32(0),
                };
                ecx.write_scalar(res, dest)?;
            }
            "pthread_threadid_np" => {
                let [thread, tid_ptr] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(libc::pthread_t, *u64) -> i32),
                    (link_name, abi, args),
                )?;
                let res = ecx.apple_pthread_threadid_np(thread, tid_ptr)?;
                ecx.write_scalar(res, dest)?;
            }

            // Synchronization primitives
            "os_sync_wait_on_address" => {
                let [addr_op, value_op, size_op, flags_op] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(*_, u64, usize, u32) -> i32),
                    (link_name, abi, args),
                )?;
                ecx.os_sync_wait_on_address(
                    addr_op,
                    value_op,
                    size_op,
                    flags_op,
                    MacOsFutexTimeout::None,
                    dest,
                )?;
            }
            "os_sync_wait_on_address_with_deadline" => {
                let [addr_op, value_op, size_op, flags_op, clock_op, timeout_op] = &ecx
                    .check_shim_sig(
                        shim_sig!(extern "C" fn(*_, u64, usize, u32, u32, u64) -> i32),
                        (link_name, abi, args),
                    )?;
                ecx.os_sync_wait_on_address(
                    addr_op,
                    value_op,
                    size_op,
                    flags_op,
                    MacOsFutexTimeout::Absolute { clock_op, timeout_op },
                    dest,
                )?;
            }
            "os_sync_wait_on_address_with_timeout" => {
                let [addr_op, value_op, size_op, flags_op, clock_op, timeout_op] = &ecx
                    .check_shim_sig(
                        shim_sig!(extern "C" fn(*_, u64, usize, u32, u32, u64) -> i32),
                        (link_name, abi, args),
                    )?;
                ecx.os_sync_wait_on_address(
                    addr_op,
                    value_op,
                    size_op,
                    flags_op,
                    MacOsFutexTimeout::Relative { clock_op, timeout_op },
                    dest,
                )?;
            }
            "os_sync_wake_by_address_any" => {
                let [addr_op, size_op, flags_op] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(*_, usize, u32) -> i32),
                    (link_name, abi, args),
                )?;
                ecx.os_sync_wake_by_address(
                    addr_op, size_op, flags_op, /* all */ false, dest,
                )?;
            }
            "os_sync_wake_by_address_all" => {
                let [addr_op, size_op, flags_op] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(*_, usize, u32) -> i32),
                    (link_name, abi, args),
                )?;
                ecx.os_sync_wake_by_address(addr_op, size_op, flags_op, /* all */ true, dest)?;
            }
            "os_unfair_lock_lock" => {
                let [lock_op] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(*libc::os_unfair_lock_s) -> ()),
                    (link_name, abi, args),
                )?;
                ecx.os_unfair_lock_lock(lock_op)?;
            }
            "os_unfair_lock_trylock" => {
                let [lock_op] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(*libc::os_unfair_lock_s) -> bool),
                    (link_name, abi, args),
                )?;
                ecx.os_unfair_lock_trylock(lock_op, dest)?;
            }
            "os_unfair_lock_unlock" => {
                let [lock_op] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(*libc::os_unfair_lock_s) -> ()),
                    (link_name, abi, args),
                )?;
                ecx.os_unfair_lock_unlock(lock_op)?;
            }
            "os_unfair_lock_assert_owner" => {
                let [lock_op] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(*libc::os_unfair_lock_s) -> ()),
                    (link_name, abi, args),
                )?;
                ecx.os_unfair_lock_assert_owner(lock_op)?;
            }
            "os_unfair_lock_assert_not_owner" => {
                let [lock_op] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(*libc::os_unfair_lock_s) -> ()),
                    (link_name, abi, args),
                )?;
                ecx.os_unfair_lock_assert_not_owner(lock_op)?;
            }
            "dispatch_semaphore_create" => {
                let [value] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(isize) -> *_),
                    (link_name, abi, args),
                )?;
                ecx.dispatch_semaphore_create(value, dest)?;
            }
            "dispatch_semaphore_wait" => {
                let [sem, timeout] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(*_, u64) -> isize),
                    (link_name, abi, args),
                )?;
                ecx.dispatch_semaphore_wait(sem, timeout, dest)?;
            }
            "dispatch_semaphore_signal" => {
                let [sem] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(*_) -> isize),
                    (link_name, abi, args),
                )?;
                ecx.dispatch_semaphore_signal(sem, dest)?;
            }
            "dispatch_release" => {
                let [sem] = &ecx
                    .check_shim_sig(shim_sig!(extern "C" fn(*_) -> ()), (link_name, abi, args))?;
                ecx.dispatch_release(sem)?;
            }
            "dispatch_time" => {
                let [when, delta] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(u64, i64) -> u64),
                    (link_name, abi, args),
                )?;
                ecx.dispatch_time(when, delta, dest)?;
            }

            "pthread_cond_timedwait_relative_np" => {
                let [cond, mutex, reltime] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(*libc::pthread_cond_t, *libc::pthread_mutex_t, *libc::timespec) -> i32),
                    (link_name, abi, args),
                )?;
                ecx.pthread_cond_timedwait(
                    cond, mutex, reltime, dest, /* macos_relative_np */ true,
                )?;
            }

            // Incomplete shims that we "stub out" just to get pre-main initialization code to work.
            // These shims are enabled only when the caller is in the standard library.
            "confstr" => {
                // FIXME: This does not have a direct test (#3179).
                let [_key, _buf, _buflen] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(i32, *_, usize) -> usize),
                    (link_name, abi, args),
                )?;
                // We just pretend that no configuration key exists, and return EINVAL.
                ecx.set_last_error(LibcError("EINVAL"))?;
                ecx.write_int(0, dest)?;
            }

            _ => return interp_ok(EmulateItemResult::NotSupported),
        };

        interp_ok(EmulateItemResult::NeedsReturn)
    }
}
