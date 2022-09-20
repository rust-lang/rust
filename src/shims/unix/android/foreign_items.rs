use rustc_span::Symbol;
use rustc_target::spec::abi::Abi;

use crate::*;
use shims::foreign_items::EmulateByNameResult;

impl<'mir, 'tcx: 'mir> EvalContextExt<'mir, 'tcx> for crate::MiriInterpCx<'mir, 'tcx> {}

pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriInterpCxExt<'mir, 'tcx> {
    fn emulate_foreign_item_by_name(
        &mut self,
        link_name: Symbol,
        _abi: Abi,
        _args: &[OpTy<'tcx, Provenance>],
        _dest: &PlaceTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx, EmulateByNameResult<'mir, 'tcx>> {
        let _this = self.eval_context_mut();
        #[allow(clippy::match_single_binding)]
        match link_name.as_str() {
            _ => return Ok(EmulateByNameResult::NotSupported),
        }

        #[allow(unreachable_code)]
        Ok(EmulateByNameResult::NeedsJumping)
    }
}
