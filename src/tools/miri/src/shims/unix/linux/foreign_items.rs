use rustc_middle::ty::Ty;
use rustc_span::Symbol;
use rustc_target::callconv::FnAbi;
use rustc_target::spec::Target;

use self::shims::unix::linux::mem::EvalContextExt as _;
use self::shims::unix::linux_like::eventfd::EvalContextExt as _;
use self::shims::unix::linux_like::syscall::syscall;
use crate::machine::{SIGRTMAX, SIGRTMIN};
use crate::shims::unix::foreign_items::EvalContextExt as _;
use crate::shims::unix::linux_like::epoll::EvalContextExt as _;
use crate::shims::unix::linux_like::thread::prctl;
use crate::shims::unix::*;
use crate::*;

// The documentation of glibc complains that the kernel never exposes
// TASK_COMM_LEN through the headers, so it's assumed to always be 16 bytes
// long including a null terminator.
const TASK_COMM_LEN: u64 = 16;

pub fn is_dyn_sym(name: &str, target: &Target) -> bool {
    matches!(name, "gettid" | "statx")
        || (target.env == rustc_target::spec::Env::Gnu
            && matches!(name, "__cxa_thread_atexit_impl"))
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
            // File related shims
            "open64" => {
                // FIXME: This does not have a direct test (#3179).
                // "open64" means that the file is opened with O_LARGEFILE, which we always do.
                let ([path_raw, flag], varargs) = ecx.check_shim_sig_variadic(
                    shim_sig!(extern "C" fn(*_, i32, ...) -> i32),
                    (link_name, abi, args),
                )?;
                let result = ecx.open(None, &path_raw, &flag, varargs)?;
                ecx.write_scalar(result, dest)?;
            }
            "openat64" => {
                // FIXME: This does not have a direct test (#3179).
                // "openat64" means that the file is opened with O_LARGEFILE, which we always do.
                let ([dirfd, path_raw, flag], varargs) = ecx.check_shim_sig_variadic(
                    shim_sig!(extern "C" fn(i32, *_, i32, ...) -> i32),
                    (link_name, abi, args),
                )?;
                let result = ecx.open(Some(&dirfd), &path_raw, &flag, varargs)?;
                ecx.write_scalar(result, dest)?;
            }
            "pread64" => {
                let [fd, buf, count, offset] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(i32, *_, usize, libc::off64_t) -> isize),
                    (link_name, abi, args),
                )?;
                let fd = ecx.read_scalar(fd)?.to_i32()?;
                let buf = ecx.read_pointer(buf)?;
                let count = ecx.read_target_usize(count)?;
                let offset = ecx.read_scalar(offset)?.to_int(offset.layout.size)?;
                ecx.read(fd, buf, count, Some(offset), dest)?;
            }
            "pwrite64" => {
                let [fd, buf, n, offset] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(i32, *_, usize, libc::off64_t) -> isize),
                    (link_name, abi, args),
                )?;
                let fd = ecx.read_scalar(fd)?.to_i32()?;
                let buf = ecx.read_pointer(buf)?;
                let count = ecx.read_target_usize(n)?;
                let offset = ecx.read_scalar(offset)?.to_int(offset.layout.size)?;
                trace!("Called pwrite64({:?}, {:?}, {:?}, {:?})", fd, buf, count, offset);
                ecx.write(fd, buf, count, Some(offset), dest)?;
            }
            "lseek64" => {
                let [fd, offset, whence] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(i32, libc::off64_t, i32) -> libc::off64_t),
                    (link_name, abi, args),
                )?;
                let fd = ecx.read_scalar(fd)?.to_i32()?;
                let offset = ecx.read_scalar(offset)?.to_int(offset.layout.size)?;
                let whence = ecx.read_scalar(whence)?.to_i32()?;
                ecx.lseek(fd, offset, whence, dest)?;
            }
            "posix_fallocate64" => {
                let [fd, offset, len] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(i32, libc::off64_t, libc::off64_t) -> i32),
                    (link_name, abi, args),
                )?;

                let fd = ecx.read_scalar(fd)?.to_i32()?;
                let offset = ecx.read_scalar(offset)?.to_i64()?;
                let len = ecx.read_scalar(len)?.to_i64()?;

                let result = ecx.posix_fallocate(fd, offset, len)?;
                ecx.write_scalar(result, dest)?;
            }
            "fallocate" => {
                let [fd, mode, offset, len] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(i32, i32, libc::off_t, libc::off_t) -> i32),
                    (link_name, abi, args),
                )?;

                let fd = ecx.read_scalar(fd)?.to_i32()?;
                let mode = ecx.read_scalar(mode)?.to_i32()?;
                // We don't support platforms which have libc::off_t bigger than 64 bits.
                let offset =
                    i64::try_from(ecx.read_scalar(offset)?.to_int(offset.layout.size)?).unwrap();
                let len = i64::try_from(ecx.read_scalar(len)?.to_int(len.layout.size)?).unwrap();

                let result = ecx.linux_fallocate(fd, mode, offset, len)?;
                ecx.write_scalar(result, dest)?;
            }
            "fallocate64" => {
                let [fd, mode, offset, len] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(i32, i32, libc::off64_t, libc::off64_t) -> i32),
                    (link_name, abi, args),
                )?;

                let fd = ecx.read_scalar(fd)?.to_i32()?;
                let mode = ecx.read_scalar(mode)?.to_i32()?;
                let offset = ecx.read_scalar(offset)?.to_i64()?;
                let len = ecx.read_scalar(len)?.to_i64()?;

                let result = ecx.linux_fallocate(fd, mode, offset, len)?;
                ecx.write_scalar(result, dest)?;
            }
            "ftruncate64" => {
                let [fd, length] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(i32, libc::off64_t) -> i32),
                    (link_name, abi, args),
                )?;
                let fd = ecx.read_scalar(fd)?.to_i32()?;
                let length = ecx.read_scalar(length)?.to_int(length.layout.size)?;
                let result = ecx.ftruncate64(fd, length)?;
                ecx.write_scalar(result, dest)?;
            }
            "sync_file_range" => {
                let [fd, offset, nbytes, flags] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(i32, libc::off64_t, libc::off64_t, u32) -> i32),
                    (link_name, abi, args),
                )?;
                let result = ecx.sync_file_range(fd, offset, nbytes, flags)?;
                ecx.write_scalar(result, dest)?;
            }
            "statx" => {
                let [dirfd, pathname, flags, mask, statxbuf] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(i32, *_, i32, u32, *libc::statx) -> i32),
                    (link_name, abi, args),
                )?;
                let result = ecx.linux_statx(dirfd, pathname, flags, mask, statxbuf)?;
                ecx.write_scalar(result, dest)?;
            }
            "fstat64" => {
                let [fd, buf] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(i32, *libc::stat64) -> i32),
                    (link_name, abi, args),
                )?;
                let result = ecx.fstat(fd, buf)?;
                ecx.write_scalar(result, dest)?;
            }
            "readdir64" => {
                let [dirp] = &ecx
                    .check_shim_sig(shim_sig!(extern "C" fn(*_) -> *_), (link_name, abi, args))?;
                ecx.readdir(dirp, dest, "dirent64")?;
            }

            // epoll, eventfd
            "epoll_create1" => {
                let [flag] = &ecx
                    .check_shim_sig(shim_sig!(extern "C" fn(i32) -> i32), (link_name, abi, args))?;
                let result = ecx.epoll_create1(flag)?;
                ecx.write_scalar(result, dest)?;
            }
            "epoll_ctl" => {
                let [epfd, op, fd, event] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(i32, i32, i32, *libc::epoll_event) -> i32),
                    (link_name, abi, args),
                )?;
                let result = ecx.epoll_ctl(epfd, op, fd, event)?;
                ecx.write_scalar(result, dest)?;
            }
            "epoll_wait" => {
                let [epfd, events, maxevents, timeout] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(i32, *libc::epoll_event, i32, i32) -> i32),
                    (link_name, abi, args),
                )?;
                ecx.epoll_wait(epfd, events, maxevents, timeout, dest)?;
            }
            "eventfd" => {
                let [val, flag] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(u32, i32) -> i32),
                    (link_name, abi, args),
                )?;
                let result = ecx.eventfd(val, flag)?;
                ecx.write_scalar(result, dest)?;
            }

            // Threading
            "pthread_setname_np" => {
                let [thread, name] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(libc::pthread_t, *_) -> i32),
                    (link_name, abi, args),
                )?;
                let res = match ecx.pthread_setname_np(
                    ecx.read_scalar(thread)?,
                    ecx.read_scalar(name)?,
                    TASK_COMM_LEN,
                    /* truncate */ false,
                )? {
                    ThreadNameResult::Ok => Scalar::from_u32(0),
                    ThreadNameResult::NameTooLong => ecx.eval_libc("ERANGE"),
                };
                ecx.write_scalar(res, dest)?;
            }
            "pthread_getname_np" => {
                let [thread, name, len] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(libc::pthread_t, *_, usize) -> i32),
                    (link_name, abi, args),
                )?;
                // The function's behavior isn't portable between platforms.
                // In case of glibc, the length of the output buffer must
                // be not shorter than TASK_COMM_LEN.
                let len = ecx.read_scalar(len)?;
                let res = if len.to_target_usize(ecx)? >= TASK_COMM_LEN {
                    match ecx.pthread_getname_np(
                        ecx.read_scalar(thread)?,
                        ecx.read_scalar(name)?,
                        len,
                        /* truncate*/ false,
                    )? {
                        ThreadNameResult::Ok => Scalar::from_u32(0),
                        ThreadNameResult::NameTooLong => unreachable!(),
                    }
                } else {
                    ecx.eval_libc("ERANGE")
                };
                ecx.write_scalar(res, dest)?;
            }
            "gettid" => {
                let [] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn() -> libc::pid_t),
                    (link_name, abi, args),
                )?;
                let result = ecx.unix_gettid(link_name.as_str())?;
                ecx.write_scalar(result, dest)?;
            }
            "prctl" => prctl(ecx, link_name, abi, args, dest)?,
            "__cxa_thread_atexit_impl"
                if ecx.tcx.sess.target.env == rustc_target::spec::Env::Gnu =>
            {
                let [dtor, arg, dso] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(*_, *_, *_) -> i32),
                    (link_name, abi, args),
                )?;
                let dtor = ecx.read_pointer(dtor)?;
                let arg = ecx.read_scalar(arg)?;
                let dso = ecx.read_pointer(dso)?;

                let dso_handle = Symbol::intern("__dso_handle");
                if dso.addr() != Pointer::from(ecx.machine.extern_statics[&dso_handle]).addr() {
                    throw_unsup_format!(
                        "__cxa_thread_atexit_impl: a DSO value other than the address of `__dso_handle` is not supported"
                    );
                }

                let dtor = ecx.get_ptr_fn(dtor)?.as_instance()?;
                ecx.machine.tls.add_thread_atexit(
                    ecx.active_thread(),
                    dtor,
                    arg,
                    ecx.machine.current_user_relevant_span(),
                )?;

                ecx.write_null(dest)?;
            }

            // Dynamically invoked syscalls
            "syscall" => {
                syscall(ecx, link_name, abi, args, dest)?;
            }

            // Miscellaneous
            "mmap64" => {
                let [addr, length, prot, flags, fd, offset] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(*_, usize, i32, i32, i32, libc::off64_t) -> *_),
                    (link_name, abi, args),
                )?;
                let offset = ecx.read_scalar(offset)?.to_i64()?;
                let ptr = ecx.mmap(addr, length, prot, flags, fd, offset.into())?;
                ecx.write_scalar(ptr, dest)?;
            }
            "mremap" => {
                let ([old_address, old_size, new_size, flags], _) = ecx.check_shim_sig_variadic(
                    shim_sig!(extern "C" fn(*_, usize, usize, i32, ...) -> *_),
                    (link_name, abi, args),
                )?;
                let ptr = ecx.mremap(&old_address, &old_size, &new_size, &flags)?;
                ecx.write_scalar(ptr, dest)?;
            }
            "__xpg_strerror_r" => {
                let [errnum, buf, buflen] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(i32, *_, usize) -> i32),
                    (link_name, abi, args),
                )?;
                let result = ecx.strerror_r(errnum, buf, buflen)?;
                ecx.write_scalar(result, dest)?;
            }
            "__errno_location" => {
                let [] =
                    &ecx.check_shim_sig(shim_sig!(extern "C" fn() -> *_), (link_name, abi, args))?;
                let errno_place = ecx.last_error_place()?;
                ecx.write_scalar(errno_place.to_ref(ecx).to_scalar(), dest)?;
            }
            "__libc_current_sigrtmin" => {
                let [] =
                    &ecx.check_shim_sig(shim_sig!(extern "C" fn() -> i32), (link_name, abi, args))?;

                ecx.write_int(SIGRTMIN, dest)?;
            }
            "__libc_current_sigrtmax" => {
                let [] =
                    &ecx.check_shim_sig(shim_sig!(extern "C" fn() -> i32), (link_name, abi, args))?;

                ecx.write_int(SIGRTMAX, dest)?;
            }

            // Incomplete shims that we "stub out" just to get pre-main initialization code to work.
            // These shims are enabled only when the caller is in the standard library.
            "pthread_getattr_np" if ecx.frame_in_std() => {
                let [_thread, _attr] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(libc::pthread_t, *_) -> i32),
                    (link_name, abi, args),
                )?;
                ecx.write_null(dest)?;
            }
            "gnu_get_libc_version"
                if ecx.frame_in_std()
                    && ecx.tcx.sess.target.env == rustc_target::spec::Env::Gnu =>
            {
                let [] =
                    &ecx.check_shim_sig(shim_sig!(extern "C" fn() -> *_), (link_name, abi, args))?;
                // We have to be at least version 2.26 so that std does not call `res_init`.
                // This returns a C string, so we have to add a null terminator.
                let version = "2.26\0";
                let version = ecx.allocate_str_dedup(version)?;
                ecx.write_pointer(version.ptr(), dest)?;
            }

            _ => return interp_ok(EmulateItemResult::NotSupported),
        };

        interp_ok(EmulateItemResult::NeedsReturn)
    }
}
