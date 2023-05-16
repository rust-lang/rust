//! This pass removes storage markers if they won't be emitted during codegen.

use crate::MirPass;
use rustc_index::bit_set::BitSet;
use rustc_middle::mir::*;
use rustc_middle::ty::TyCtxt;
use rustc_mir_dataflow::impls::borrowed_locals;

pub struct RemoveStorageMarkers;

impl<'tcx> MirPass<'tcx> for RemoveStorageMarkers {
    fn is_enabled(&self, sess: &rustc_session::Session) -> bool {
        sess.mir_opt_level() > 0
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        let storage_to_keep = if tcx.sess.emit_lifetime_markers() {
            borrowed_locals(body)
        } else {
            BitSet::new_empty(body.local_decls.len())
        };

        trace!("Running RemoveStorageMarkers on {:?}", body.source);
        for data in body.basic_blocks.as_mut_preserves_cfg() {
            data.statements.retain(|statement| match statement.kind {
                StatementKind::StorageLive(local) | StatementKind::StorageDead(local) => {
                    storage_to_keep.contains(local)
                }
                StatementKind::Nop => false,
                _ => true,
            })
        }
    }
}
