use rustc_middle::ty::Ty;
use rustc_span::Symbol;
use rustc_target::callconv::FnAbi;

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
            // Environment
            "__unsetenv13" => {
                let [name] = &ecx
                    .check_shim_sig(shim_sig!(extern "C" fn(*_) -> i32), (link_name, abi, args))?;
                let result = ecx.unsetenv(name)?;
                ecx.write_scalar(result, dest)?;
            }

            // Miscellaneous
            "__errno" => {
                let [] =
                    &ecx.check_shim_sig(shim_sig!(extern "C" fn() -> *_), (link_name, abi, args))?;
                let errno_place = ecx.last_error_place()?;
                ecx.write_scalar(errno_place.to_ref(ecx).to_scalar(), dest)?;
            }

            _ => return interp_ok(EmulateItemResult::NotSupported),
        }
        interp_ok(EmulateItemResult::NeedsReturn)
    }
}
