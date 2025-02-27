use std::mem;

use rustc_middle::mir::*;
use rustc_middle::ty::TyCtxt;
use tracing::{debug, instrument, trace};

pub(super) struct BranchDuplicator;

impl<'tcx> crate::MirPass<'tcx> for BranchDuplicator {
    fn is_enabled(&self, sess: &rustc_session::Session) -> bool {
        sess.mir_opt_level() >= 2
    }

    #[instrument(skip_all level = "debug")]
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        let def_id = body.source.def_id();
        debug!(?def_id);

        // Optimizing coroutines creates query cycles.
        if tcx.is_coroutine(def_id) {
            trace!("Skipped for coroutine {:?}", def_id);
            return;
        }

        let is_branch = |targets: &SwitchTargets| {
            targets.all_targets().len() == 2
                || (targets.all_values().len() == 2
                    && body.basic_blocks[targets.otherwise()].is_empty_unreachable())
        };

        let mut candidates = Vec::new();
        for (bb, bbdata) in body.basic_blocks.iter_enumerated() {
            if let TerminatorKind::SwitchInt { targets, .. } = &bbdata.terminator().kind
                && is_branch(targets)
                && let Ok(preds) =
                    <[BasicBlock; 2]>::try_from(body.basic_blocks.predecessors()[bb].as_slice())
                && preds.iter().copied().all(|p| {
                    matches!(body.basic_blocks[p].terminator().kind, TerminatorKind::Goto { .. })
                })
                && bbdata.statements.iter().all(|x| is_negligible(&x.kind))
            {
                candidates.push((bb, preds));
            }
        }

        if candidates.is_empty() {
            return;
        }

        let basic_blocks = body.basic_blocks.as_mut();
        for (bb, [p0, p1]) in candidates {
            let bbdata = &mut basic_blocks[bb];
            let statements = mem::take(&mut bbdata.statements);
            let unreachable = Terminator {
                source_info: bbdata.terminator().source_info,
                kind: TerminatorKind::Unreachable,
            };
            let terminator = mem::replace(bbdata.terminator_mut(), unreachable);

            let pred0data = &mut basic_blocks[p0];
            pred0data.statements.extend(statements.iter().cloned());
            *pred0data.terminator_mut() = terminator.clone();

            let pred1data = &mut basic_blocks[p1];
            pred1data.statements.extend(statements);
            *pred1data.terminator_mut() = terminator;
        }
    }

    fn is_required(&self) -> bool {
        false
    }
}

fn is_negligible<'tcx>(stmt: &StatementKind<'tcx>) -> bool {
    use Rvalue::*;
    use StatementKind::*;
    match stmt {
        StorageLive(..) | StorageDead(..) => true,
        Assign(place_and_rvalue) => match &place_and_rvalue.1 {
            Ref(..) | RawPtr(..) | Discriminant(..) | NullaryOp(..) => true,
            _ => false,
        },
        _ => false,
    }
}
