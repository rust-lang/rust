use rustc_index::bit_set::DenseBitSet;
use rustc_middle::mir::*;
use rustc_middle::ty::TyCtxt;
use rustc_mir_dataflow::{Analysis, GenKill};

// use super::simplify::simplify_cfg;

pub(crate) struct RemoveDeadDrops;

use rustc_middle::mir::visit::*;

struct MaybeInitializedLocals;

/// This is conservative: If any part of a local is initialized, we mark it
/// initialized, while we only mark uninitialized if the whole local is moved
/// from or StorageDead.
struct TransferFunction<'a, T> {
    trans: &'a mut T,
}

impl<'tcx, T: GenKill<Local>> Visitor<'tcx> for TransferFunction<'_, T> {
    fn visit_place(&mut self, place: &Place<'tcx>, context: PlaceContext, location: Location) {
        self.super_place(place, context, location);

        if context.is_place_assignment() {
            self.trans.gen_(place.local);
        } else if matches!(
            context,
            PlaceContext::NonUse(NonUseContext::StorageLive | NonUseContext::StorageDead)
        ) {
            self.trans.kill(place.local);
        }
    }

    fn visit_operand(&mut self, operand: &Operand<'tcx>, location: Location) {
        self.super_operand(operand, location);

        if let Operand::Move(place) = operand {
            if place.projection.is_empty() {
                self.trans.kill(place.local);
            }
        }
    }
}

impl<'tcx> Analysis<'tcx> for MaybeInitializedLocals {
    type Domain = DenseBitSet<Local>;
    const NAME: &'static str = "maybe_initialized_locals";

    fn bottom_value(&self, body: &Body<'tcx>) -> Self::Domain {
        DenseBitSet::new_empty(body.local_decls.len())
    }

    fn initialize_start_block(&self, body: &Body<'tcx>, state: &mut Self::Domain) {
        for arg in 1..=body.arg_count {
            state.insert(Local::from_usize(arg));
        }
    }

    fn apply_primary_statement_effect(
        &self,
        state: &mut Self::Domain,
        stmt: &Statement<'tcx>,
        location: Location,
    ) {
        TransferFunction { trans: state }.visit_statement(stmt, location);
    }

    fn apply_primary_terminator_effect<'mir>(
        &self,
        state: &mut Self::Domain,
        terminator: &'mir Terminator<'tcx>,
        location: Location,
    ) -> TerminatorEdges<'mir, 'tcx> {
        TransferFunction { trans: state }.visit_terminator(terminator, location);
        terminator.edges()
    }
}

impl<'tcx> crate::MirPass<'tcx> for RemoveDeadDrops {
    fn is_required(&self) -> bool {
        true
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        let mut maybe_init_cursor = MaybeInitializedLocals
            .iterate_to_fixpoint(tcx, body, None)
            .into_results_cursor(body);

        let mut dead_drops = Vec::new();

        for (block, data) in body.basic_blocks.iter_enumerated() {
            if let Some(terminator) = &data.terminator
                && let TerminatorKind::Drop { place, target, .. } = &terminator.kind
            {
                let term_location = Location { block, statement_index: data.statements.len() };
                maybe_init_cursor.seek_before_primary_effect(term_location);

                let is_dead = !maybe_init_cursor.get().contains(place.local);
                if is_dead {
                    dead_drops.push((block, *target));
                }
            }
        }

        if !dead_drops.is_empty() {
            for (block, target) in dead_drops {
                if let Some(terminator) = &mut body.basic_blocks.as_mut()[block].terminator {
                    terminator.kind = TerminatorKind::Goto { target };
                }
            }

            // Removing drop terminators may simplify the CFG, so run cleanup.
            // simplify_cfg(tcx, body);
        }
    }
}
