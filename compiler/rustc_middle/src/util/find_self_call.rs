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
    {
        debug!("find_self_call: func={:?}", func);
        if let Operand::Constant(box ConstOperand { const_, .. }) = func {
            if let ty::FnDef(def_id, fn_args) = *const_.ty().kind() {
                if let Some(ty::AssocItem { fn_has_self_parameter: true, .. }) =
                    tcx.opt_associated_item(def_id)
                {
                    debug!("find_self_call: args={:?}", fn_args);
                    if let [
                        Spanned {
                            node: Operand::Move(self_place) | Operand::Copy(self_place), ..
                        },
                        ..,
                    ] = **args
                    {
                        if self_place.as_local() == Some(local) {
                            return Some((def_id, fn_args));
                        }
                    }
                }
            }
        }
    }
    None
}
