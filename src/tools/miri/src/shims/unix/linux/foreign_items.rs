use rustc_abi::CanonAbi;
use rustc_middle::ty::Ty;
use rustc_span::Symbol;
use rustc_target::callconv::FnAbi;

use self::shims::unix::linux::mem::EvalContextExt as _;
use self::shims::unix::linux_like::epoll::EvalContextExt as _;
use self::shims::unix::linux_like::eventfd::EvalContextExt as _;
use self::shims::unix::linux_like::syscall::syscall;
use crate::machine::{SIGRTMAX, SIGRTMIN};
use crate::shims::unix::foreign_items::EvalContextExt as _;
use crate::shims::unix::*;
use crate::*;

// The documentation of glibc complains that the kernel never exposes
// TASK_COMM_LEN through the headers, so it's assumed to always be 16 bytes
// long including a null terminator.
const TASK_COMM_LEN: u64 = 16;

pub fn is_dyn_sym(name: &str) -> bool {
    matches!(name, "statx")
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
            // File related shims
            "readdir64" => {
                let [dirp] = this.check_shim(abi, CanonAbi::C, link_name, args)?;
                let result = this.linux_solarish_readdir64("dirent64", dirp)?;
                this.write_scalar(result, dest)?;
            }
            "sync_file_range" => {
                let [fd, offset, nbytes, flags] =
                    this.check_shim(abi, CanonAbi::C, link_name, args)?;
                let result = this.sync_file_range(fd, offset, nbytes, flags)?;
                this.write_scalar(result, dest)?;
            }
            "statx" => {
                let [dirfd, pathname, flags, mask, statxbuf] =
                    this.check_shim(abi, CanonAbi::C, link_name, args)?;
                let result = this.linux_statx(dirfd, pathname, flags, mask, statxbuf)?;
                this.write_scalar(result, dest)?;
            }

            // epoll, eventfd
            "epoll_create1" => {
                let [flag] = this.check_shim(abi, CanonAbi::C, link_name, args)?;
                let result = this.epoll_create1(flag)?;
                this.write_scalar(result, dest)?;
            }
            "epoll_ctl" => {
                let [epfd, op, fd, event] = this.check_shim(abi, CanonAbi::C, link_name, args)?;
                let result = this.epoll_ctl(epfd, op, fd, event)?;
                this.write_scalar(result, dest)?;
            }
            "epoll_wait" => {
                let [epfd, events, maxevents, timeout] =
                    this.check_shim(abi, CanonAbi::C, link_name, args)?;
                this.epoll_wait(epfd, events, maxevents, timeout, dest)?;
            }
            "eventfd" => {
                let [val, flag] = this.check_shim(abi, CanonAbi::C, link_name, args)?;
                let result = this.eventfd(val, flag)?;
                this.write_scalar(result, dest)?;
            }

            // Threading
            "pthread_setname_np" => {
                let [thread, name] = this.check_shim(abi, CanonAbi::C, link_name, args)?;
                let res = match this.pthread_setname_np(
                    this.read_scalar(thread)?,
                    this.read_scalar(name)?,
                    TASK_COMM_LEN,
                    /* truncate */ false,
                )? {
                    ThreadNameResult::Ok => Scalar::from_u32(0),
                    ThreadNameResult::NameTooLong => this.eval_libc("ERANGE"),
                    // Act like we faild to open `/proc/self/task/$tid/comm`.
                    ThreadNameResult::ThreadNotFound => this.eval_libc("ENOENT"),
                };
                this.write_scalar(res, dest)?;
            }
            "pthread_getname_np" => {
                let [thread, name, len] = this.check_shim(abi, CanonAbi::C, link_name, args)?;
                // The function's behavior isn't portable between platforms.
                // In case of glibc, the length of the output buffer must
                // be not shorter than TASK_COMM_LEN.
                let len = this.read_scalar(len)?;
                let res = if len.to_target_usize(this)? >= TASK_COMM_LEN {
                    match this.pthread_getname_np(
                        this.read_scalar(thread)?,
                        this.read_scalar(name)?,
                        len,
                        /* truncate*/ false,
                    )? {
                        ThreadNameResult::Ok => Scalar::from_u32(0),
                        ThreadNameResult::NameTooLong => unreachable!(),
                        // Act like we faild to open `/proc/self/task/$tid/comm`.
                        ThreadNameResult::ThreadNotFound => this.eval_libc("ENOENT"),
                    }
                } else {
                    this.eval_libc("ERANGE")
                };
                this.write_scalar(res, dest)?;
            }
            "gettid" => {
                let [] = this.check_shim(abi, CanonAbi::C, link_name, args)?;
                let result = this.linux_gettid()?;
                this.write_scalar(result, dest)?;
            }

            // Dynamically invoked syscalls
            "syscall" => {
                syscall(this, link_name, abi, args, dest)?;
            }

            // Miscellaneous
            "mmap64" => {
                let [addr, length, prot, flags, fd, offset] =
                    this.check_shim(abi, CanonAbi::C, link_name, args)?;
                let offset = this.read_scalar(offset)?.to_i64()?;
                let ptr = this.mmap(addr, length, prot, flags, fd, offset.into())?;
                this.write_scalar(ptr, dest)?;
            }
            "mremap" => {
                let ([old_address, old_size, new_size, flags], _) =
                    this.check_shim_variadic(abi, CanonAbi::C, link_name, args)?;
                let ptr = this.mremap(old_address, old_size, new_size, flags)?;
                this.write_scalar(ptr, dest)?;
            }
            "__xpg_strerror_r" => {
                let [errnum, buf, buflen] = this.check_shim(abi, CanonAbi::C, link_name, args)?;
                let result = this.strerror_r(errnum, buf, buflen)?;
                this.write_scalar(result, dest)?;
            }
            "__errno_location" => {
                let [] = this.check_shim(abi, CanonAbi::C, link_name, args)?;
                let errno_place = this.last_error_place()?;
                this.write_scalar(errno_place.to_ref(this).to_scalar(), dest)?;
            }
            "__libc_current_sigrtmin" => {
                let [] = this.check_shim(abi, CanonAbi::C, link_name, args)?;

                this.write_int(SIGRTMIN, dest)?;
            }
            "__libc_current_sigrtmax" => {
                let [] = this.check_shim(abi, CanonAbi::C, link_name, args)?;

                this.write_int(SIGRTMAX, dest)?;
            }

            // Incomplete shims that we "stub out" just to get pre-main initialization code to work.
            // These shims are enabled only when the caller is in the standard library.
            "pthread_getattr_np" if this.frame_in_std() => {
                let [_thread, _attr] = this.check_shim(abi, CanonAbi::C, link_name, args)?;
                this.write_null(dest)?;
            }

            _ => return interp_ok(EmulateItemResult::NotSupported),
        };

        interp_ok(EmulateItemResult::NeedsReturn)
    }
}
