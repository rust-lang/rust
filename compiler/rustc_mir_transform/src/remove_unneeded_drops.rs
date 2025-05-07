//! This pass replaces a drop of a type that does not need dropping, with a goto.
//!
//! When the MIR is built, we check `needs_drop` before emitting a `Drop` for a place. This pass is
//! useful because (unlike MIR building) it runs after type checking, so it can make use of
//! `TypingMode::PostAnalysis` to provide more precise type information, especially about opaque
//! types.

use rustc_middle::mir::*;
use rustc_middle::ty::TyCtxt;
use tracing::{debug, trace};

use super::simplify::simplify_cfg;

pub(super) struct RemoveUnneededDrops;

impl<'tcx> crate::MirPass<'tcx> for RemoveUnneededDrops {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        trace!("Running RemoveUnneededDrops on {:?}", body.source);

        let typing_env = body.typing_env(tcx);
        let mut should_simplify = false;
        for block in body.basic_blocks.as_mut() {
            let terminator = block.terminator_mut();
            if let TerminatorKind::Drop { place, target, .. } = terminator.kind {
                let ty = place.ty(&body.local_decls, tcx);
                if ty.ty.needs_drop(tcx, typing_env) {
                    continue;
                }
                debug!("SUCCESS: replacing `drop` with goto({:?})", target);
                terminator.kind = TerminatorKind::Goto { target };
                should_simplify = true;
            }
        }

        // if we applied optimizations, we potentially have some cfg to cleanup to
        // make it easier for further passes
        if should_simplify {
            simplify_cfg(tcx, body);
        }
    }

    fn is_required(&self) -> bool {
        true
    }
}
