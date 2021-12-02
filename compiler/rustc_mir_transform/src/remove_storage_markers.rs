//! This pass removes storage markers if they won't be emitted during codegen.

use crate::MirPass;
use rustc_middle::mir::*;
use rustc_middle::ty::TyCtxt;

pub struct RemoveStorageMarkers;

impl<'tcx> MirPass<'tcx> for RemoveStorageMarkers {
    fn is_enabled(&self, sess: &rustc_session::Session) -> bool {
        sess.mir_opt_level() > 0
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        if tcx.sess.emit_lifetime_markers() {
            return;
        }

        trace!("Running RemoveStorageMarkers on {:?}", body.source);
        for data in body.basic_blocks_mut() {
            data.statements.retain(|statement| match statement.kind {
                StatementKind::StorageLive(..)
                | StatementKind::StorageDead(..)
                | StatementKind::Nop => false,
                _ => true,
            })
        }
    }
}
