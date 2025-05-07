use rustc_middle::ty::Ty;
use rustc_span::Symbol;
use rustc_target::callconv::{Conv, FnAbi};

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
        let this = self.eval_context_mut();

        // See `fn emulate_foreign_item_inner` in `shims/foreign_items.rs` for the general pattern.

        match link_name.as_str() {
            // errno
            "__error" => {
                let [] = this.check_shim(abi, Conv::C, link_name, args)?;
                let errno_place = this.last_error_place()?;
                this.write_scalar(errno_place.to_ref(this).to_scalar(), dest)?;
            }

            // File related shims
            "close$NOCANCEL" => {
                let [result] = this.check_shim(abi, Conv::C, link_name, args)?;
                let result = this.close(result)?;
                this.write_scalar(result, dest)?;
            }
            "stat" | "stat64" | "stat$INODE64" => {
                let [path, buf] = this.check_shim(abi, Conv::C, link_name, args)?;
                let result = this.macos_fbsd_solarish_stat(path, buf)?;
                this.write_scalar(result, dest)?;
            }
            "lstat" | "lstat64" | "lstat$INODE64" => {
                let [path, buf] = this.check_shim(abi, Conv::C, link_name, args)?;
                let result = this.macos_fbsd_solarish_lstat(path, buf)?;
                this.write_scalar(result, dest)?;
            }
            "fstat" | "fstat64" | "fstat$INODE64" => {
                let [fd, buf] = this.check_shim(abi, Conv::C, link_name, args)?;
                let result = this.macos_fbsd_solarish_fstat(fd, buf)?;
                this.write_scalar(result, dest)?;
            }
            "opendir$INODE64" => {
                let [name] = this.check_shim(abi, Conv::C, link_name, args)?;
                let result = this.opendir(name)?;
                this.write_scalar(result, dest)?;
            }
            "readdir_r" | "readdir_r$INODE64" => {
                let [dirp, entry, result] = this.check_shim(abi, Conv::C, link_name, args)?;
                let result = this.macos_fbsd_readdir_r(dirp, entry, result)?;
                this.write_scalar(result, dest)?;
            }
            "realpath$DARWIN_EXTSN" => {
                let [path, resolved_path] = this.check_shim(abi, Conv::C, link_name, args)?;
                let result = this.realpath(path, resolved_path)?;
                this.write_scalar(result, dest)?;
            }
            "ioctl" => {
                let ([fd_num, cmd], varargs) =
                    this.check_shim_variadic(abi, Conv::C, link_name, args)?;
                let result = this.ioctl(fd_num, cmd, varargs)?;
                this.write_scalar(result, dest)?;
            }

            // Environment related shims
            "_NSGetEnviron" => {
                let [] = this.check_shim(abi, Conv::C, link_name, args)?;
                let environ = this.machine.env_vars.unix().environ();
                this.write_pointer(environ, dest)?;
            }

            // Random data generation
            "CCRandomGenerateBytes" => {
                let [bytes, count] = this.check_shim(abi, Conv::C, link_name, args)?;
                let bytes = this.read_pointer(bytes)?;
                let count = this.read_target_usize(count)?;
                let success = this.eval_libc_i32("kCCSuccess");
                this.gen_random(bytes, count)?;
                this.write_int(success, dest)?;
            }

            // Time related shims
            "mach_absolute_time" => {
                let [] = this.check_shim(abi, Conv::C, link_name, args)?;
                let result = this.mach_absolute_time()?;
                this.write_scalar(result, dest)?;
            }

            "mach_timebase_info" => {
                let [info] = this.check_shim(abi, Conv::C, link_name, args)?;
                let result = this.mach_timebase_info(info)?;
                this.write_scalar(result, dest)?;
            }

            // Access to command-line arguments
            "_NSGetArgc" => {
                let [] = this.check_shim(abi, Conv::C, link_name, args)?;
                this.write_pointer(this.machine.argc.expect("machine must be initialized"), dest)?;
            }
            "_NSGetArgv" => {
                let [] = this.check_shim(abi, Conv::C, link_name, args)?;
                this.write_pointer(this.machine.argv.expect("machine must be initialized"), dest)?;
            }
            "_NSGetExecutablePath" => {
                let [buf, bufsize] = this.check_shim(abi, Conv::C, link_name, args)?;
                this.check_no_isolation("`_NSGetExecutablePath`")?;

                let buf_ptr = this.read_pointer(buf)?;
                let bufsize = this.deref_pointer_as(bufsize, this.machine.layouts.u32)?;

                // Using the host current_exe is a bit off, but consistent with Linux
                // (where stdlib reads /proc/self/exe).
                let path = std::env::current_exe().unwrap();
                let (written, size_needed) = this.write_path_to_c_str(
                    &path,
                    buf_ptr,
                    this.read_scalar(&bufsize)?.to_u32()?.into(),
                )?;

                if written {
                    this.write_null(dest)?;
                } else {
                    this.write_scalar(Scalar::from_u32(size_needed.try_into().unwrap()), &bufsize)?;
                    this.write_int(-1, dest)?;
                }
            }

            // Thread-local storage
            "_tlv_atexit" => {
                let [dtor, data] = this.check_shim(abi, Conv::C, link_name, args)?;
                let dtor = this.read_pointer(dtor)?;
                let dtor = this.get_ptr_fn(dtor)?.as_instance()?;
                let data = this.read_scalar(data)?;
                let active_thread = this.active_thread();
                this.machine.tls.add_macos_thread_dtor(active_thread, dtor, data)?;
            }

            // Querying system information
            "pthread_get_stackaddr_np" => {
                let [thread] = this.check_shim(abi, Conv::C, link_name, args)?;
                this.read_target_usize(thread)?;
                let stack_addr = Scalar::from_uint(this.machine.stack_addr, this.pointer_size());
                this.write_scalar(stack_addr, dest)?;
            }
            "pthread_get_stacksize_np" => {
                let [thread] = this.check_shim(abi, Conv::C, link_name, args)?;
                this.read_target_usize(thread)?;
                let stack_size = Scalar::from_uint(this.machine.stack_size, this.pointer_size());
                this.write_scalar(stack_size, dest)?;
            }

            // Threading
            "pthread_setname_np" => {
                let [name] = this.check_shim(abi, Conv::C, link_name, args)?;

                // The real implementation has logic in two places:
                // * in userland at https://github.com/apple-oss-distributions/libpthread/blob/c032e0b076700a0a47db75528a282b8d3a06531a/src/pthread.c#L1178-L1200,
                // * in kernel at https://github.com/apple-oss-distributions/xnu/blob/8d741a5de7ff4191bf97d57b9f54c2f6d4a15585/bsd/kern/proc_info.c#L3218-L3227.
                //
                // The function in libc calls the kernel to validate
                // the security policies and the input. If all of the requirements
                // are met, then the name is set and 0 is returned. Otherwise, if
                // the specified name is lomnger than MAXTHREADNAMESIZE, then
                // ENAMETOOLONG is returned.
                let thread = this.pthread_self()?;
                let res = match this.pthread_setname_np(
                    thread,
                    this.read_scalar(name)?,
                    this.eval_libc("MAXTHREADNAMESIZE").to_target_usize(this)?.try_into().unwrap(),
                    /* truncate */ false,
                )? {
                    ThreadNameResult::Ok => Scalar::from_u32(0),
                    ThreadNameResult::NameTooLong => this.eval_libc("ENAMETOOLONG"),
                    ThreadNameResult::ThreadNotFound => unreachable!(),
                };
                // Contrary to the manpage, `pthread_setname_np` on macOS still
                // returns an integer indicating success.
                this.write_scalar(res, dest)?;
            }
            "pthread_getname_np" => {
                let [thread, name, len] = this.check_shim(abi, Conv::C, link_name, args)?;

                // The function's behavior isn't portable between platforms.
                // In case of macOS, a truncated name (due to a too small buffer)
                // does not lead to an error.
                //
                // For details, see the implementation at
                // https://github.com/apple-oss-distributions/libpthread/blob/c032e0b076700a0a47db75528a282b8d3a06531a/src/pthread.c#L1160-L1175.
                // The key part is the strlcpy, which truncates the resulting value,
                // but always null terminates (except for zero sized buffers).
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

            // Synchronization primitives
            "os_sync_wait_on_address" => {
                let [addr_op, value_op, size_op, flags_op] =
                    this.check_shim(abi, Conv::C, link_name, args)?;
                this.os_sync_wait_on_address(
                    addr_op,
                    value_op,
                    size_op,
                    flags_op,
                    MacOsFutexTimeout::None,
                    dest,
                )?;
            }
            "os_sync_wait_on_address_with_deadline" => {
                let [addr_op, value_op, size_op, flags_op, clock_op, timeout_op] =
                    this.check_shim(abi, Conv::C, link_name, args)?;
                this.os_sync_wait_on_address(
                    addr_op,
                    value_op,
                    size_op,
                    flags_op,
                    MacOsFutexTimeout::Absolute { clock_op, timeout_op },
                    dest,
                )?;
            }
            "os_sync_wait_on_address_with_timeout" => {
                let [addr_op, value_op, size_op, flags_op, clock_op, timeout_op] =
                    this.check_shim(abi, Conv::C, link_name, args)?;
                this.os_sync_wait_on_address(
                    addr_op,
                    value_op,
                    size_op,
                    flags_op,
                    MacOsFutexTimeout::Relative { clock_op, timeout_op },
                    dest,
                )?;
            }
            "os_sync_wake_by_address_any" => {
                let [addr_op, size_op, flags_op] =
                    this.check_shim(abi, Conv::C, link_name, args)?;
                this.os_sync_wake_by_address(
                    addr_op, size_op, flags_op, /* all */ false, dest,
                )?;
            }
            "os_sync_wake_by_address_all" => {
                let [addr_op, size_op, flags_op] =
                    this.check_shim(abi, Conv::C, link_name, args)?;
                this.os_sync_wake_by_address(
                    addr_op, size_op, flags_op, /* all */ true, dest,
                )?;
            }
            "os_unfair_lock_lock" => {
                let [lock_op] = this.check_shim(abi, Conv::C, link_name, args)?;
                this.os_unfair_lock_lock(lock_op)?;
            }
            "os_unfair_lock_trylock" => {
                let [lock_op] = this.check_shim(abi, Conv::C, link_name, args)?;
                this.os_unfair_lock_trylock(lock_op, dest)?;
            }
            "os_unfair_lock_unlock" => {
                let [lock_op] = this.check_shim(abi, Conv::C, link_name, args)?;
                this.os_unfair_lock_unlock(lock_op)?;
            }
            "os_unfair_lock_assert_owner" => {
                let [lock_op] = this.check_shim(abi, Conv::C, link_name, args)?;
                this.os_unfair_lock_assert_owner(lock_op)?;
            }
            "os_unfair_lock_assert_not_owner" => {
                let [lock_op] = this.check_shim(abi, Conv::C, link_name, args)?;
                this.os_unfair_lock_assert_not_owner(lock_op)?;
            }

            _ => return interp_ok(EmulateItemResult::NotSupported),
        };

        interp_ok(EmulateItemResult::NeedsReturn)
    }

    fn ioctl(
        &mut self,
        fd_num: &OpTy<'tcx>,
        cmd: &OpTy<'tcx>,
        _varargs: &[OpTy<'tcx>],
    ) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_mut();

        let fioclex = this.eval_libc_u64("FIOCLEX");

        let fd_num = this.read_scalar(fd_num)?.to_i32()?;
        let cmd = this.read_scalar(cmd)?.to_u64()?;

        if cmd == fioclex {
            // Since we don't support `exec`, this is a NOP. However, we want to
            // return EBADF if the FD is invalid.
            if this.machine.fds.is_fd_num(fd_num) {
                interp_ok(Scalar::from_i32(0))
            } else {
                this.set_last_error_and_return_i32(LibcError("EBADF"))
            }
        } else {
            throw_unsup_format!("ioctl: unsupported command {cmd:#x}");
        }
    }
}
