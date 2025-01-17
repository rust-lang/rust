use rustc_span::def_id::DefId;
use rustc_span::source_map::Spanned;
use tracing::debug;

use crate::mir::*;
use crate::ty::{self, GenericArgsRef, TyCtxt};

/// Checks if the specified `local` is used as the `self` parameter of a method call
/// in the provided `BasicBlock`. If it is, then the `DefId` of the called method is
/// returned.
pub fn find_self_call<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    local: Local,
    block: BasicBlock,
) -> Option<(DefId, GenericArgsRef<'tcx>)> {
    debug!("find_self_call(local={:?}): terminator={:?}", local, body[block].terminator);
    if let Some(Terminator { kind: TerminatorKind::Call { func, args, .. }, .. }) =
        &body[block].terminator
        && let Operand::Constant(box ConstOperand { const_, .. }) = func
        && let ty::FnDef(def_id, fn_args) = *const_.ty().kind()
        && let Some(ty::AssocItem { fn_has_self_parameter: true, .. }) =
            tcx.opt_associated_item(def_id)
        && let [Spanned { node: Operand::Move(self_place) | Operand::Copy(self_place), .. }, ..] =
            **args
    {
        if self_place.as_local() == Some(local) {
            return Some((def_id, fn_args));
        }

        // Handle the case where `self_place` gets reborrowed.
        // This happens when the receiver is `&T`.
        for stmt in &body[block].statements {
            if let StatementKind::Assign(box (place, rvalue)) = &stmt.kind
                && let Some(reborrow_local) = place.as_local()
                && self_place.as_local() == Some(reborrow_local)
                && let Rvalue::Ref(_, _, deref_place) = rvalue
                && let PlaceRef { local: deref_local, projection: [ProjectionElem::Deref] } =
                    deref_place.as_ref()
                && deref_local == local
            {
                return Some((def_id, fn_args));
            }
        }
    }
    None
}
