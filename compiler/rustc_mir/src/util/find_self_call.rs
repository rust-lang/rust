use rustc_middle::mir::*;
use rustc_middle::ty::subst::SubstsRef;
use rustc_middle::ty::{self, TyCtxt};
use rustc_span::def_id::DefId;

/// Checks if the specified `local` is used as the `self` parameter of a method call
/// in the provided `BasicBlock`. If it is, then the `DefId` of the called method is
/// returned.
pub fn find_self_call<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    local: Local,
    block: BasicBlock,
) -> Option<(DefId, SubstsRef<'tcx>)> {
    debug!("find_self_call(local={:?}): terminator={:?}", local, &body[block].terminator);
    if let Some(Terminator { kind: TerminatorKind::Call { func, args, .. }, .. }) =
        &body[block].terminator
    {
        debug!("find_self_call: func={:?}", func);
        if let Operand::Constant(box Constant { literal, .. }) = func {
            if let ty::FnDef(def_id, substs) = *literal.ty().kind() {
                if let Some(ty::AssocItem { fn_has_self_parameter: true, .. }) =
                    tcx.opt_associated_item(def_id)
                {
                    debug!("find_self_call: args={:?}", args);
                    if let [Operand::Move(self_place) | Operand::Copy(self_place), ..] = **args {
                        if self_place.as_local() == Some(local) {
                            return Some((def_id, substs));
                        }
                    }
                }
            }
        }
    }
    None
}
