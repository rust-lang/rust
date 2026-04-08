use rustc_middle::mir::*;
use rustc_middle::ty::TyCtxt;
use rustc_mir_dataflow::impls::MaybeInitializedPlaces;
use rustc_mir_dataflow::move_paths::{LookupResult, MoveData};
use rustc_mir_dataflow::{Analysis, MaybeReachable};

pub(crate) struct RemoveDeadDrops;

impl<'tcx> crate::MirPass<'tcx> for RemoveDeadDrops {
    fn is_required(&self) -> bool {
        true
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        if body.coroutine.is_none() {
            return;
        }

        let move_data = MoveData::gather_moves(body, tcx, |_| true);

        let mut maybe_init_cursor = MaybeInitializedPlaces::new(tcx, body, &move_data)
            .iterate_to_fixpoint(tcx, body, None)
            .into_results_cursor(body);

        let mut dead_drops = Vec::new();

        for (block, data) in body.basic_blocks.iter_enumerated() {
            if let Some(terminator) = &data.terminator
                && let TerminatorKind::Drop { place, target, .. } = &terminator.kind
            {
                let LookupResult::Exact(path) = move_data.rev_lookup.find(place.as_ref()) else {
                    continue;
                };

                let term_location = Location { block, statement_index: data.statements.len() };
                maybe_init_cursor.seek_before_primary_effect(term_location);

                let is_dead = match maybe_init_cursor.get() {
                    MaybeReachable::Unreachable => true,
                    MaybeReachable::Reachable(maybe_init) => !maybe_init.contains(path),
                };

                if is_dead {
                    dead_drops.push((block, *target));
                }
            }
        }

        for (block, target) in dead_drops {
            if let Some(terminator) = &mut body.basic_blocks.as_mut()[block].terminator {
                terminator.kind = TerminatorKind::Goto { target };
            }
        }
    }
}
