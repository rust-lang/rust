//! Looks for basic blocks that are just a simple 2-way branch,
//! and replicates them into their predecessors.
//!
//! That sounds wasteful, but it's very common that the predecessor just set
//! whatever we're about to branch on, and this makes that easier to collapse
//! later. And if not, well, an extra branch is no big deal.

use crate::MirPass;

use rustc_middle::mir::*;
use rustc_middle::ty::TyCtxt;

use super::simplify::simplify_cfg;

pub struct ReplicateBranches;

impl<'tcx> MirPass<'tcx> for ReplicateBranches {
    fn is_enabled(&self, sess: &rustc_session::Session) -> bool {
        sess.mir_opt_level() >= 2
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        debug!("Running ReplicateBranches on `{:?}`", body.source);

        let mut source_and_target_blocks = vec![];

        for (bb, data) in body.basic_blocks.iter_enumerated() {
            if data.is_cleanup {
                continue;
            }

            let TerminatorKind::SwitchInt { discr, targets } = &data.terminator().kind
            else { continue };

            let Some(place) = discr.place() else { continue };

            // Only replicate simple branches. That means either:
            // - A specific target plus a potentially-reachable otherwise, or
            // - Two specific targets and the otherwise is unreachable.
            if !(targets.iter().len() < 2
                || (targets.iter().len() <= 2
                    && body.basic_blocks[targets.otherwise()].is_empty_unreachable()))
            {
                continue;
            }

            // Only replicate blocks that branch on a discriminant.
            let mut found_discriminant = false;
            for stmt in &data.statements {
                match &stmt.kind {
                    StatementKind::StorageDead { .. } | StatementKind::StorageLive { .. } => {}
                    StatementKind::Assign(place_and_rvalue)
                        if place_and_rvalue.0 == place
                            && matches!(place_and_rvalue.1, Rvalue::Discriminant(..)) =>
                    {
                        found_discriminant = true;
                    }
                    _ => continue,
                }
            }

            // This currently doesn't duplicate ordinary boolean checks as that regresses
            // various tests that seem to depend on the existing structure.
            if !found_discriminant {
                continue;
            }

            // Only replicate to a small number of `goto` predecessors.
            let preds = &body.basic_blocks.predecessors()[bb];
            if preds.len() > 2
                || !preds.iter().copied().all(|p| {
                    let pdata = &body.basic_blocks[p];
                    matches!(pdata.terminator().kind, TerminatorKind::Goto { .. })
                        && !pdata.is_cleanup
                })
            {
                continue;
            }

            for p in preds {
                source_and_target_blocks.push((*p, bb));
            }
        }

        if source_and_target_blocks.is_empty() {
            return;
        }

        let basic_blocks = body.basic_blocks.as_mut();
        for (source, target) in source_and_target_blocks {
            debug_assert_eq!(
                basic_blocks[source].terminator().kind,
                TerminatorKind::Goto { target },
            );

            let (source, target) = basic_blocks.pick2_mut(source, target);
            source.statements.extend(target.statements.iter().cloned());
            source.terminator = target.terminator.clone();
        }

        // The target blocks should be unused now, so cleanup after ourselves.
        simplify_cfg(tcx, body);
    }
}
