use rustc_abi::CanonAbi;
use rustc_middle::ty::Ty;
use rustc_span::Symbol;
use rustc_target::callconv::FnAbi;

use crate::shims::unix::android::thread::prctl;
use crate::shims::unix::env::EvalContextExt as _;
use crate::shims::unix::linux_like::epoll::EvalContextExt as _;
use crate::shims::unix::linux_like::eventfd::EvalContextExt as _;
use crate::shims::unix::linux_like::syscall::syscall;
use crate::shims::unix::*;
use crate::*;

pub fn is_dyn_sym(name: &str) -> bool {
    matches!(name, "gettid")
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
            // File related shims
            "stat" => {
                let [path, buf] = this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                let result = this.stat(path, buf)?;
                this.write_scalar(result, dest)?;
            }
            "lstat" => {
                let [path, buf] = this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                let result = this.lstat(path, buf)?;
                this.write_scalar(result, dest)?;
            }
            "readdir" => {
                let [dirp] = this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                let result = this.readdir64("dirent", dirp)?;
                this.write_scalar(result, dest)?;
            }
            "pread64" => {
                let [fd, buf, count, offset] = this.check_shim_sig(
                    shim_sig!(extern "C" fn(i32, *mut _, usize, libc::off64_t) -> isize),
                    link_name,
                    abi,
                    args,
                )?;
                let fd = this.read_scalar(fd)?.to_i32()?;
                let buf = this.read_pointer(buf)?;
                let count = this.read_target_usize(count)?;
                let offset = this.read_scalar(offset)?.to_int(offset.layout.size)?;
                this.read(fd, buf, count, Some(offset), dest)?;
            }
            "pwrite64" => {
                let [fd, buf, n, offset] = this.check_shim_sig(
                    shim_sig!(extern "C" fn(i32, *const _, usize, libc::off64_t) -> isize),
                    link_name,
                    abi,
                    args,
                )?;
                let fd = this.read_scalar(fd)?.to_i32()?;
                let buf = this.read_pointer(buf)?;
                let count = this.read_target_usize(n)?;
                let offset = this.read_scalar(offset)?.to_int(offset.layout.size)?;
                trace!("Called pwrite64({:?}, {:?}, {:?}, {:?})", fd, buf, count, offset);
                this.write(fd, buf, count, Some(offset), dest)?;
            }
            "lseek64" => {
                let [fd, offset, whence] = this.check_shim_sig(
                    shim_sig!(extern "C" fn(i32, libc::off64_t, i32) -> libc::off64_t),
                    link_name,
                    abi,
                    args,
                )?;
                let fd = this.read_scalar(fd)?.to_i32()?;
                let offset = this.read_scalar(offset)?.to_int(offset.layout.size)?;
                let whence = this.read_scalar(whence)?.to_i32()?;
                this.lseek64(fd, offset, whence, dest)?;
            }
            "ftruncate64" => {
                let [fd, length] = this.check_shim_sig(
                    shim_sig!(extern "C" fn(i32, libc::off64_t) -> i32),
                    link_name,
                    abi,
                    args,
                )?;
                let fd = this.read_scalar(fd)?.to_i32()?;
                let length = this.read_scalar(length)?.to_int(length.layout.size)?;
                let result = this.ftruncate64(fd, length)?;
                this.write_scalar(result, dest)?;
            }

            // epoll, eventfd
            "epoll_create1" => {
                let [flag] = this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                let result = this.epoll_create1(flag)?;
                this.write_scalar(result, dest)?;
            }
            "epoll_ctl" => {
                let [epfd, op, fd, event] =
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                let result = this.epoll_ctl(epfd, op, fd, event)?;
                this.write_scalar(result, dest)?;
            }
            "epoll_wait" => {
                let [epfd, events, maxevents, timeout] =
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                this.epoll_wait(epfd, events, maxevents, timeout, dest)?;
            }
            "eventfd" => {
                let [val, flag] = this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                let result = this.eventfd(val, flag)?;
                this.write_scalar(result, dest)?;
            }

            // Miscellaneous
            "__errno" => {
                let [] = this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                let errno_place = this.last_error_place()?;
                this.write_scalar(errno_place.to_ref(this).to_scalar(), dest)?;
            }

            "gettid" => {
                let [] = this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                let result = this.unix_gettid(link_name.as_str())?;
                this.write_scalar(result, dest)?;
            }

            // Dynamically invoked syscalls
            "syscall" => syscall(this, link_name, abi, args, dest)?,

            // Threading
            "prctl" => prctl(this, link_name, abi, args, dest)?,

            _ => return interp_ok(EmulateItemResult::NotSupported),
        }
        interp_ok(EmulateItemResult::NeedsReturn)
    }
}
