use rustc_abi::ExternAbi;
use rustc_span::Symbol;

use self::shims::unix::linux::epoll::EvalContextExt as _;
use self::shims::unix::linux::eventfd::EvalContextExt as _;
use crate::shims::unix::android::thread::prctl;
use crate::shims::unix::linux::syscall::syscall;
use crate::*;

pub fn is_dyn_sym(_name: &str) -> bool {
    false
}

impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
pub trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    fn emulate_foreign_item_inner(
        &mut self,
        link_name: Symbol,
        abi: ExternAbi,
        args: &[OpTy<'tcx>],
        dest: &MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx, EmulateItemResult> {
        let this = self.eval_context_mut();
        match link_name.as_str() {
            // epoll, eventfd
            "epoll_create1" => {
                let [flag] =
                    this.check_shim(abi, ExternAbi::C { unwind: false }, link_name, args)?;
                let result = this.epoll_create1(flag)?;
                this.write_scalar(result, dest)?;
            }
            "epoll_ctl" => {
                let [epfd, op, fd, event] =
                    this.check_shim(abi, ExternAbi::C { unwind: false }, link_name, args)?;
                let result = this.epoll_ctl(epfd, op, fd, event)?;
                this.write_scalar(result, dest)?;
            }
            "epoll_wait" => {
                let [epfd, events, maxevents, timeout] =
                    this.check_shim(abi, ExternAbi::C { unwind: false }, link_name, args)?;
                this.epoll_wait(epfd, events, maxevents, timeout, dest)?;
            }
            "eventfd" => {
                let [val, flag] =
                    this.check_shim(abi, ExternAbi::C { unwind: false }, link_name, args)?;
                let result = this.eventfd(val, flag)?;
                this.write_scalar(result, dest)?;
            }

            // Miscellaneous
            "__errno" => {
                let [] = this.check_shim(abi, ExternAbi::C { unwind: false }, link_name, args)?;
                let errno_place = this.last_error_place()?;
                this.write_scalar(errno_place.to_ref(this).to_scalar(), dest)?;
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
