use rustc_span::Symbol;
use rustc_target::spec::abi::Abi;

use crate::*;
use shims::EmulateItemResult;

pub fn is_dyn_sym(_name: &str) -> bool {
    false
}

impl<'mir, 'tcx: 'mir> EvalContextExt<'mir, 'tcx> for crate::MiriInterpCx<'mir, 'tcx> {}
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriInterpCxExt<'mir, 'tcx> {
    #[allow(warnings)]
    fn emulate_foreign_item_inner(
        &mut self,
        link_name: Symbol,
        abi: Abi,
        args: &[OpTy<'tcx, Provenance>],
        dest: &MPlaceTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx, EmulateItemResult> {
        let this = self.eval_context_mut();
        match link_name.as_str() {
            _ => return Ok(EmulateItemResult::NotSupported),
        }
        Ok(EmulateItemResult::NeedsJumping)
    }
}
