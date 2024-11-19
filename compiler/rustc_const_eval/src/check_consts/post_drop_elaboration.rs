use rustc_middle::mir::visit::Visitor;
use rustc_middle::mir::{self, BasicBlock, Location};
use rustc_middle::ty::TyCtxt;
use rustc_span::symbol::sym;
use tracing::trace;

use super::ConstCx;
use super::check::Qualifs;
use super::ops::{self};
use super::qualifs::{NeedsNonConstDrop, Qualif};
use crate::check_consts::check::Checker;
use crate::check_consts::qualifs::NeedsDrop;
use crate::check_consts::rustc_allow_const_fn_unstable;

/// Returns `true` if we should use the more precise live drop checker that runs after drop
/// elaboration.
pub fn checking_enabled(ccx: &ConstCx<'_, '_>) -> bool {
    // Const-stable functions must always use the stable live drop checker...
    if ccx.enforce_recursive_const_stability() {
        // ...except if they have the feature flag set via `rustc_allow_const_fn_unstable`.
        return rustc_allow_const_fn_unstable(
            ccx.tcx,
            ccx.body.source.def_id().expect_local(),
            sym::const_precise_live_drops,
        );
    }

    ccx.tcx.features().const_precise_live_drops()
}

/// Look for live drops in a const context.
///
/// This is separate from the rest of the const checking logic because it must run after drop
/// elaboration.
pub fn check_live_drops<'tcx>(tcx: TyCtxt<'tcx>, body: &mir::Body<'tcx>) {
    let ccx = ConstCx::new(tcx, body);
    if ccx.const_kind.is_none() {
        return;
    }

    if tcx.has_attr(body.source.def_id(), sym::rustc_do_not_const_check) {
        return;
    }

    if !checking_enabled(&ccx) {
        return;
    }

    let mut visitor = CheckLiveDrops { ccx: &ccx, qualifs: Qualifs::default() };

    visitor.visit_body(body);
}

struct CheckLiveDrops<'mir, 'tcx> {
    ccx: &'mir ConstCx<'mir, 'tcx>,
    qualifs: Qualifs<'mir, 'tcx>,
}

// So we can access `body` and `tcx`.
impl<'mir, 'tcx> std::ops::Deref for CheckLiveDrops<'mir, 'tcx> {
    type Target = ConstCx<'mir, 'tcx>;

    fn deref(&self) -> &Self::Target {
        self.ccx
    }
}

impl<'tcx> Visitor<'tcx> for CheckLiveDrops<'_, 'tcx> {
    fn visit_basic_block_data(&mut self, bb: BasicBlock, block: &mir::BasicBlockData<'tcx>) {
        trace!("visit_basic_block_data: bb={:?} is_cleanup={:?}", bb, block.is_cleanup);

        // Ignore drop terminators in cleanup blocks.
        if block.is_cleanup {
            return;
        }

        self.super_basic_block_data(bb, block);
    }

    fn visit_terminator(&mut self, terminator: &mir::Terminator<'tcx>, location: Location) {
        trace!("visit_terminator: terminator={:?} location={:?}", terminator, location);

        match &terminator.kind {
            mir::TerminatorKind::Drop { place: dropped_place, .. } => {
                let ty_of_dropped_place = dropped_place.ty(self.body, self.tcx).ty;

                let needs_drop = if let Some(local) = dropped_place.as_local() {
                    self.qualifs.needs_drop(self.ccx, local, location)
                } else {
                    NeedsDrop::in_any_value_of_ty(self.ccx, ty_of_dropped_place)
                };
                // If this type doesn't need a drop at all, then there's nothing to enforce.
                if !needs_drop {
                    return;
                }

                let mut err_span = terminator.source_info.span;

                let needs_non_const_drop = if let Some(local) = dropped_place.as_local() {
                    // Use the span where the local was declared as the span of the drop error.
                    err_span = self.body.local_decls[local].source_info.span;
                    self.qualifs.needs_non_const_drop(self.ccx, local, location)
                } else {
                    NeedsNonConstDrop::in_any_value_of_ty(self.ccx, ty_of_dropped_place)
                };

                // I know it's not great to be creating a new const checker, but I'd
                // rather use it so we can deduplicate the error emitting logic that
                // it contains.
                Checker::new(self.ccx).check_op_spanned_post(
                    ops::LiveDrop {
                        dropped_at: Some(terminator.source_info.span),
                        dropped_ty: ty_of_dropped_place,
                        needs_non_const_drop,
                    },
                    err_span,
                );
            }

            mir::TerminatorKind::UnwindTerminate(_)
            | mir::TerminatorKind::Call { .. }
            | mir::TerminatorKind::TailCall { .. }
            | mir::TerminatorKind::Assert { .. }
            | mir::TerminatorKind::FalseEdge { .. }
            | mir::TerminatorKind::FalseUnwind { .. }
            | mir::TerminatorKind::CoroutineDrop
            | mir::TerminatorKind::Goto { .. }
            | mir::TerminatorKind::InlineAsm { .. }
            | mir::TerminatorKind::UnwindResume
            | mir::TerminatorKind::Return
            | mir::TerminatorKind::SwitchInt { .. }
            | mir::TerminatorKind::Unreachable
            | mir::TerminatorKind::Yield { .. } => {}
        }
    }
}
