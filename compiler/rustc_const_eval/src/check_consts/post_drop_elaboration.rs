use rustc_middle::mir::visit::Visitor;
use rustc_middle::mir::{self, BasicBlock, Location};
use rustc_middle::ty::TyCtxt;
use rustc_span::sym;
use tracing::trace;

use super::ConstCx;
use crate::check_consts::check::Checker;
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

    // I know it's not great to be creating a new const checker, but I'd
    // rather use it so we can deduplicate the error emitting logic that
    // it contains.
    let mut visitor = CheckLiveDrops { checker: Checker::new(&ccx) };

    visitor.visit_body(body);
}

struct CheckLiveDrops<'mir, 'tcx> {
    checker: Checker<'mir, 'tcx>,
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
                self.checker.check_drop_terminator(
                    *dropped_place,
                    location,
                    terminator.source_info.span,
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
