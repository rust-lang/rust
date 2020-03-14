//! The `RemoveUnitStorage` pass removes `StorageLive` and `StorageDead` statements
//! which operates on locals of type `()`.

use crate::transform::{MirPass, MirSource};
use rustc::mir::*;
use rustc::ty::TyCtxt;

pub struct RemoveUnitStorage;

impl<'tcx> MirPass<'tcx> for RemoveUnitStorage {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, _: MirSource<'tcx>, body: &mut BodyAndCache<'tcx>) {
        // This pass removes UB, so only run it when optimizations are enabled.
        if tcx.sess.opts.debugging_opts.mir_opt_level == 0 {
            return;
        }

        let (blocks, locals) = body.basic_blocks_and_local_decls_mut();

        for block in blocks {
            for stmt in &mut block.statements {
                if let StatementKind::StorageLive(l) | StatementKind::StorageDead(l) = stmt.kind {
                    if locals[l].ty == tcx.types.unit {
                        stmt.make_nop();
                    }
                }
            }
        }
    }
}
