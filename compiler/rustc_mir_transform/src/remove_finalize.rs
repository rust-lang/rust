//! Removes Finalize statements, as they have no codegen impact

use crate::MirPass;
use rustc_middle::mir::{Body, StatementKind};
use rustc_middle::ty::TyCtxt;

pub struct RemoveFinalize;

impl<'tcx> MirPass<'tcx> for RemoveFinalize {
    fn run_pass(&self, _tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        for block in body.basic_blocks_mut() {
            block
                .statements
                .retain(|statement| !matches!(statement.kind, StatementKind::Finalize(..)));
        }
    }
}
