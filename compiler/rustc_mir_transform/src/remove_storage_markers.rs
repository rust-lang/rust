//! This pass removes storage markers if they won't be emitted during codegen.

use rustc_middle::mir::*;
use rustc_middle::ty::TyCtxt;
use tracing::trace;

pub(super) struct RemoveStorageMarkers;

impl<'tcx> crate::MirPass<'tcx> for RemoveStorageMarkers {
    fn is_enabled(&self, sess: &rustc_session::Session) -> bool {
        sess.mir_opt_level() > 0 && !sess.emit_lifetime_markers()
    }

    fn run_pass(&self, _tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        trace!("Running RemoveStorageMarkers on {:?}", body.source);
        for data in body.basic_blocks.as_mut_preserves_cfg() {
            data.retain_statements(|statement| match statement.kind {
                StatementKind::StorageLive(..)
                | StatementKind::StorageDead(..)
                | StatementKind::Nop => false,
                _ => true,
            })
        }
    }

    fn is_required(&self) -> bool {
        true
    }
}
