use rustc_abi::ExternAbi;
use rustc_span::Symbol;

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
