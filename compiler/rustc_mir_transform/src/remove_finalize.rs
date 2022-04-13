//! Removes assignments to ZST places.

use crate::MirPass;
use rustc_middle::mir::{Body, StatementKind};
use rustc_middle::ty::TyCtxt;

pub struct RemoveFinalize;

impl<'tcx> MirPass<'tcx> for RemoveFinalize {
    fn is_enabled(&self, sess: &rustc_session::Session) -> bool {
        sess.mir_opt_level() > 0
    }

    fn run_pass(&self, _tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        for block in body.basic_blocks_mut() {
            block
                .statements
                .retain(|statement| !matches!(statement.kind, StatementKind::Finalize(..)));
        }
    }
}
