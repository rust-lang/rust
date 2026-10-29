use rustc_middle::ty::Ty;
use rustc_span::Symbol;
use rustc_target::callconv::FnAbi;

use crate::shims::unix::env::EvalContextExt as _;
use crate::shims::unix::linux_like::epoll::EvalContextExt as _;
use crate::shims::unix::linux_like::eventfd::EvalContextExt as _;
use crate::shims::unix::linux_like::syscall::syscall;
use crate::shims::unix::linux_like::thread::prctl;
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
        let ecx = self.eval_context_mut();
        match link_name.as_str() {
            // File related shims
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

            // Miscellaneous
            "__errno" => {
                let [] =
                    &ecx.check_shim_sig(shim_sig!(extern "C" fn() -> *_), (link_name, abi, args))?;
                let errno_place = ecx.last_error_place()?;
                ecx.write_scalar(errno_place.to_ref(ecx).to_scalar(), dest)?;
            }

            "gettid" => {
                let [] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn() -> libc::pid_t),
                    (link_name, abi, args),
                )?;
                let result = ecx.unix_gettid(link_name.as_str())?;
                ecx.write_scalar(result, dest)?;
            }

            // Dynamically invoked syscalls
            "syscall" => syscall(ecx, link_name, abi, args, dest)?,

            // Threading
            "prctl" => prctl(ecx, link_name, abi, args, dest)?,

            _ => return interp_ok(EmulateItemResult::NotSupported),
        }
        interp_ok(EmulateItemResult::NeedsReturn)
    }
}
