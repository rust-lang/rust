//! This module implements a dead store elimination (DSE) routine.
//!
//! This transformation was written specifically for the needs of dest prop. Although it is
//! perfectly sound to use it in any context that might need it, its behavior should not be changed
//! without analyzing the interaction this will have with dest prop. Specifically, in addition to
//! the soundness of this pass in general, dest prop needs it to satisfy two additional conditions:
//!
//!  1. It's idempotent, meaning that running this pass a second time immediately after running it a
//!     first time will not cause any further changes.
//!  2. This idempotence persists across dest prop's main transform, in other words inserting any
//!     number of iterations of dest prop between the first and second application of this transform
//!     will still not cause any further changes.
//!

use rustc_index::bit_set::BitSet;
use rustc_middle::mir::*;
use rustc_middle::ty::TyCtxt;
use rustc_mir_dataflow::impls::{borrowed_locals, MaybeTransitiveLiveLocals};
use rustc_mir_dataflow::Analysis;

/// Performs the optimization on the body
///
/// The `always_live` set must be a `BitSet` of all the locals that are considered always alive and
/// never eliminated. This should be, at least, the set of locals which are ever borrowed in this
/// body. It may include other locals as well if necessary. The minimum set of always alive locals
/// can be generated via the [`borrowed_locals`] function.
pub fn eliminate<'tcx>(tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>, always_live: &BitSet<Local>) {
    let mut live = MaybeTransitiveLiveLocals::new(always_live)
        .into_engine(tcx, body)
        .iterate_to_fixpoint()
        .into_results_cursor(body);

    let mut patch = Vec::new();
    for (bb, bb_data) in traversal::preorder(body) {
        for (statement_index, statement) in bb_data.statements.iter().enumerate().rev() {
            let loc = Location { block: bb, statement_index };
            if let StatementKind::Assign(assign) = &statement.kind {
                if !assign.1.is_safe_to_remove() {
                    continue;
                }
            }
            match &statement.kind {
                StatementKind::Assign(box (place, _))
                | StatementKind::SetDiscriminant { place: box place, .. }
                | StatementKind::Deinit(box place) => {
                    if !place.is_indirect() && !always_live.contains(place.local) {
                        live.seek_before_primary_effect(loc);
                        if !live.get().contains(place.local) {
                            patch.push(loc);
                        }
                    }
                }
                StatementKind::Retag(_, _)
                | StatementKind::StorageLive(_)
                | StatementKind::StorageDead(_)
                | StatementKind::Coverage(_)
                | StatementKind::Intrinsic(_)
                | StatementKind::Nop => (),

                StatementKind::FakeRead(_) | StatementKind::AscribeUserType(_, _) => {
                    bug!("{:?} not found in this MIR phase!", &statement.kind)
                }
            }
        }
    }

    if patch.is_empty() {
        return;
    }

    let bbs = body.basic_blocks.as_mut_preserves_cfg();
    for Location { block, statement_index } in patch {
        bbs[block].statements[statement_index].make_nop();
    }
}

pub struct DeadStoreElimination;

impl<'tcx> MirPass<'tcx> for DeadStoreElimination {
    fn is_enabled(&self, sess: &rustc_session::Session) -> bool {
        sess.mir_opt_level() >= 2
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        let mut always_live = borrowed_locals(body);

        // Include any locals which are used by debuginfo unless we're at a high enough MIR opt
        // level that degrading debuginfo is acceptable.
        if tcx.sess.mir_opt_level() < 3 {
            for x in &body.var_debug_info {
                match x.value {
                    VarDebugInfoContents::Place(p) => {
                        always_live.insert(p.local);
                    }
                    VarDebugInfoContents::Const(..) => {}
                }
            }
        }

        eliminate(tcx, body, &always_live);
    }
}
