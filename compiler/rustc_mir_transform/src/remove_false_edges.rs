use rustc_middle::mir::{Body, TerminatorKind};
use rustc_middle::ty::TyCtxt;

use crate::MirPass;

/// Removes `FalseEdge` and `FalseUnwind` terminators from the MIR.
///
/// These are only needed for borrow checking, and can be removed afterwards.
///
/// FIXME: This should probably have its own MIR phase.
pub struct RemoveFalseEdges;

impl<'tcx> MirPass<'tcx> for RemoveFalseEdges {
    fn run_pass(&self, _: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        for block in body.basic_blocks_mut() {
            let terminator = block.terminator_mut();
            terminator.kind = match terminator.kind {
                TerminatorKind::FalseEdge { real_target, .. } => {
                    TerminatorKind::Goto { target: real_target }
                }
                TerminatorKind::FalseUnwind { real_target, .. } => {
                    TerminatorKind::Goto { target: real_target }
                }

                _ => continue,
            }
        }
    }
}
