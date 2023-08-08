use rustc_middle::mir::{self, BasicBlock, Location, SwitchTargets, UnwindAction};
use rustc_middle::ty::TyCtxt;
use std::ops::RangeInclusive;

use super::visitor::{ResultsVisitable, ResultsVisitor};
use super::{
    Analysis, CallReturnPlaces, Effect, EffectIndex, GenKillAnalysis, GenKillSet, SwitchIntTarget,
};

pub trait Direction {
    const IS_FORWARD: bool;

    const IS_BACKWARD: bool = !Self::IS_FORWARD;

    /// Applies all effects between the given `EffectIndex`s.
    ///
    /// `effects.start()` must precede or equal `effects.end()` in this direction.
    fn apply_effects_in_range<'tcx, A>(
        analysis: &mut A,
        state: &mut A::Domain,
        block: BasicBlock,
        block_data: &mir::BasicBlockData<'tcx>,
        effects: RangeInclusive<EffectIndex>,
    ) where
        A: Analysis<'tcx>;

    fn apply_effects_in_block<'tcx, A>(
        analysis: &mut A,
        state: &mut A::Domain,
        block: BasicBlock,
        block_data: &mir::BasicBlockData<'tcx>,
    ) where
        A: Analysis<'tcx>;

    fn gen_kill_effects_in_block<'tcx, A>(
        analysis: &mut A,
        trans: &mut GenKillSet<A::Idx>,
        block: BasicBlock,
        block_data: &mir::BasicBlockData<'tcx>,
    ) where
        A: GenKillAnalysis<'tcx>;

    fn visit_results_in_block<'mir, 'tcx, F, R>(
        state: &mut F,
        block: BasicBlock,
        block_data: &'mir mir::BasicBlockData<'tcx>,
        results: &mut R,
        vis: &mut impl ResultsVisitor<'mir, 'tcx, R, FlowState = F>,
    ) where
        R: ResultsVisitable<'tcx, FlowState = F>;

    fn join_state_into_successors_of<'tcx, A>(
        analysis: &mut A,
        tcx: TyCtxt<'tcx>,
        body: &mir::Body<'tcx>,
        exit_state: &mut A::Domain,
        block: (BasicBlock, &'_ mir::BasicBlockData<'tcx>),
        propagate: impl FnMut(BasicBlock, &A::Domain),
    ) where
        A: Analysis<'tcx>;
}

/// Dataflow that runs from the exit of a block (the terminator), to its entry (the first statement).
pub struct Backward;

impl Direction for Backward {
    const IS_FORWARD: bool = false;

    fn apply_effects_in_block<'tcx, A>(
        analysis: &mut A,
        state: &mut A::Domain,
        block: BasicBlock,
        block_data: &mir::BasicBlockData<'tcx>,
    ) where
        A: Analysis<'tcx>,
    {
        let terminator = block_data.terminator();
        let location = Location { block, statement_index: block_data.statements.len() };
        analysis.apply_before_terminator_effect(state, terminator, location);
        analysis.apply_terminator_effect(state, terminator, location);

        for (statement_index, statement) in block_data.statements.iter().enumerate().rev() {
            let location = Location { block, statement_index };
            analysis.apply_before_statement_effect(state, statement, location);
            analysis.apply_statement_effect(state, statement, location);
        }
    }

    fn gen_kill_effects_in_block<'tcx, A>(
        analysis: &mut A,
        trans: &mut GenKillSet<A::Idx>,
        block: BasicBlock,
        block_data: &mir::BasicBlockData<'tcx>,
    ) where
        A: GenKillAnalysis<'tcx>,
    {
        let terminator = block_data.terminator();
        let location = Location { block, statement_index: block_data.statements.len() };
        analysis.before_terminator_effect(trans, terminator, location);
        analysis.terminator_effect(trans, terminator, location);

        for (statement_index, statement) in block_data.statements.iter().enumerate().rev() {
            let location = Location { block, statement_index };
            analysis.before_statement_effect(trans, statement, location);
            analysis.statement_effect(trans, statement, location);
        }
    }

    fn apply_effects_in_range<'tcx, A>(
        analysis: &mut A,
        state: &mut A::Domain,
        block: BasicBlock,
        block_data: &mir::BasicBlockData<'tcx>,
        effects: RangeInclusive<EffectIndex>,
    ) where
        A: Analysis<'tcx>,
    {
        let (from, to) = (*effects.start(), *effects.end());
        let terminator_index = block_data.statements.len();

        assert!(from.statement_index <= terminator_index);
        assert!(!to.precedes_in_backward_order(from));

        // Handle the statement (or terminator) at `from`.

        let next_effect = match from.effect {
            // If we need to apply the terminator effect in all or in part, do so now.
            _ if from.statement_index == terminator_index => {
                let location = Location { block, statement_index: from.statement_index };
                let terminator = block_data.terminator();

                if from.effect == Effect::Before {
                    analysis.apply_before_terminator_effect(state, terminator, location);
                    if to == Effect::Before.at_index(terminator_index) {
                        return;
                    }
                }

                analysis.apply_terminator_effect(state, terminator, location);
                if to == Effect::Primary.at_index(terminator_index) {
                    return;
                }

                // If `from.statement_index` is `0`, we will have hit one of the earlier comparisons
                // with `to`.
                from.statement_index - 1
            }

            Effect::Primary => {
                let location = Location { block, statement_index: from.statement_index };
                let statement = &block_data.statements[from.statement_index];

                analysis.apply_statement_effect(state, statement, location);
                if to == Effect::Primary.at_index(from.statement_index) {
                    return;
                }

                from.statement_index - 1
            }

            Effect::Before => from.statement_index,
        };

        // Handle all statements between `first_unapplied_idx` and `to.statement_index`.

        for statement_index in (to.statement_index..next_effect).rev().map(|i| i + 1) {
            let location = Location { block, statement_index };
            let statement = &block_data.statements[statement_index];
            analysis.apply_before_statement_effect(state, statement, location);
            analysis.apply_statement_effect(state, statement, location);
        }

        // Handle the statement at `to`.

        let location = Location { block, statement_index: to.statement_index };
        let statement = &block_data.statements[to.statement_index];
        analysis.apply_before_statement_effect(state, statement, location);

        if to.effect == Effect::Before {
            return;
        }

        analysis.apply_statement_effect(state, statement, location);
    }

    fn visit_results_in_block<'mir, 'tcx, F, R>(
        state: &mut F,
        block: BasicBlock,
        block_data: &'mir mir::BasicBlockData<'tcx>,
        results: &mut R,
        vis: &mut impl ResultsVisitor<'mir, 'tcx, R, FlowState = F>,
    ) where
        R: ResultsVisitable<'tcx, FlowState = F>,
    {
        results.reset_to_block_entry(state, block);

        vis.visit_block_end(results, &state, block_data, block);

        // Terminator
        let loc = Location { block, statement_index: block_data.statements.len() };
        let term = block_data.terminator();
        results.reconstruct_before_terminator_effect(state, term, loc);
        vis.visit_terminator_before_primary_effect(results, state, term, loc);
        results.reconstruct_terminator_effect(state, term, loc);
        vis.visit_terminator_after_primary_effect(results, state, term, loc);

        for (statement_index, stmt) in block_data.statements.iter().enumerate().rev() {
            let loc = Location { block, statement_index };
            results.reconstruct_before_statement_effect(state, stmt, loc);
            vis.visit_statement_before_primary_effect(results, state, stmt, loc);
            results.reconstruct_statement_effect(state, stmt, loc);
            vis.visit_statement_after_primary_effect(results, state, stmt, loc);
        }

        vis.visit_block_start(results, state, block_data, block);
    }

    fn join_state_into_successors_of<'tcx, A>(
        analysis: &mut A,
        _tcx: TyCtxt<'tcx>,
        body: &mir::Body<'tcx>,
        exit_state: &mut A::Domain,
        (bb, _bb_data): (BasicBlock, &'_ mir::BasicBlockData<'tcx>),
        mut propagate: impl FnMut(BasicBlock, &A::Domain),
    ) where
        A: Analysis<'tcx>,
    {
        for pred in body.basic_blocks.predecessors()[bb].iter().copied() {
            match body[pred].terminator().kind {
                // Apply terminator-specific edge effects.
                //
                // FIXME(ecstaticmorse): Avoid cloning the exit state unconditionally.
                mir::TerminatorKind::Call { destination, target: Some(dest), .. } if dest == bb => {
                    let mut tmp = exit_state.clone();
                    analysis.apply_call_return_effect(
                        &mut tmp,
                        pred,
                        CallReturnPlaces::Call(destination),
                    );
                    propagate(pred, &tmp);
                }

                mir::TerminatorKind::InlineAsm {
                    destination: Some(dest), ref operands, ..
                } if dest == bb => {
                    let mut tmp = exit_state.clone();
                    analysis.apply_call_return_effect(
                        &mut tmp,
                        pred,
                        CallReturnPlaces::InlineAsm(operands),
                    );
                    propagate(pred, &tmp);
                }

                mir::TerminatorKind::Yield { resume, resume_arg, .. } if resume == bb => {
                    let mut tmp = exit_state.clone();
                    analysis.apply_yield_resume_effect(&mut tmp, resume, resume_arg);
                    propagate(pred, &tmp);
                }

                mir::TerminatorKind::SwitchInt { targets: _, ref discr } => {
                    let mut applier = BackwardSwitchIntEdgeEffectsApplier {
                        body,
                        pred,
                        exit_state,
                        bb,
                        propagate: &mut propagate,
                        effects_applied: false,
                    };

                    analysis.apply_switch_int_edge_effects(pred, discr, &mut applier);

                    if !applier.effects_applied {
                        propagate(pred, exit_state)
                    }
                }

                _ => propagate(pred, exit_state),
            }
        }
    }
}

struct BackwardSwitchIntEdgeEffectsApplier<'a, 'tcx, D, F> {
    body: &'a mir::Body<'tcx>,
    pred: BasicBlock,
    exit_state: &'a mut D,
    bb: BasicBlock,
    propagate: &'a mut F,
    effects_applied: bool,
}

impl<D, F> super::SwitchIntEdgeEffects<D> for BackwardSwitchIntEdgeEffectsApplier<'_, '_, D, F>
where
    D: Clone,
    F: FnMut(BasicBlock, &D),
{
    fn apply(&mut self, mut apply_edge_effect: impl FnMut(&mut D, SwitchIntTarget)) {
        assert!(!self.effects_applied);

        let values = &self.body.basic_blocks.switch_sources()[&(self.bb, self.pred)];
        let targets = values.iter().map(|&value| SwitchIntTarget { value, target: self.bb });

        let mut tmp = None;
        for target in targets {
            let tmp = opt_clone_from_or_clone(&mut tmp, self.exit_state);
            apply_edge_effect(tmp, target);
            (self.propagate)(self.pred, tmp);
        }

        self.effects_applied = true;
    }
}

/// Dataflow that runs from the entry of a block (the first statement), to its exit (terminator).
pub struct Forward;

impl Direction for Forward {
    const IS_FORWARD: bool = true;

    fn apply_effects_in_block<'tcx, A>(
        analysis: &mut A,
        state: &mut A::Domain,
        block: BasicBlock,
        block_data: &mir::BasicBlockData<'tcx>,
    ) where
        A: Analysis<'tcx>,
    {
        for (statement_index, statement) in block_data.statements.iter().enumerate() {
            let location = Location { block, statement_index };
            analysis.apply_before_statement_effect(state, statement, location);
            analysis.apply_statement_effect(state, statement, location);
        }

        let terminator = block_data.terminator();
        let location = Location { block, statement_index: block_data.statements.len() };
        analysis.apply_before_terminator_effect(state, terminator, location);
        analysis.apply_terminator_effect(state, terminator, location);
    }

    fn gen_kill_effects_in_block<'tcx, A>(
        analysis: &mut A,
        trans: &mut GenKillSet<A::Idx>,
        block: BasicBlock,
        block_data: &mir::BasicBlockData<'tcx>,
    ) where
        A: GenKillAnalysis<'tcx>,
    {
        for (statement_index, statement) in block_data.statements.iter().enumerate() {
            let location = Location { block, statement_index };
            analysis.before_statement_effect(trans, statement, location);
            analysis.statement_effect(trans, statement, location);
        }

        let terminator = block_data.terminator();
        let location = Location { block, statement_index: block_data.statements.len() };
        analysis.before_terminator_effect(trans, terminator, location);
        analysis.terminator_effect(trans, terminator, location);
    }

    fn apply_effects_in_range<'tcx, A>(
        analysis: &mut A,
        state: &mut A::Domain,
        block: BasicBlock,
        block_data: &mir::BasicBlockData<'tcx>,
        effects: RangeInclusive<EffectIndex>,
    ) where
        A: Analysis<'tcx>,
    {
        let (from, to) = (*effects.start(), *effects.end());
        let terminator_index = block_data.statements.len();

        assert!(to.statement_index <= terminator_index);
        assert!(!to.precedes_in_forward_order(from));

        // If we have applied the before affect of the statement or terminator at `from` but not its
        // after effect, do so now and start the loop below from the next statement.

        let first_unapplied_index = match from.effect {
            Effect::Before => from.statement_index,

            Effect::Primary if from.statement_index == terminator_index => {
                debug_assert_eq!(from, to);

                let location = Location { block, statement_index: terminator_index };
                let terminator = block_data.terminator();
                analysis.apply_terminator_effect(state, terminator, location);
                return;
            }

            Effect::Primary => {
                let location = Location { block, statement_index: from.statement_index };
                let statement = &block_data.statements[from.statement_index];
                analysis.apply_statement_effect(state, statement, location);

                // If we only needed to apply the after effect of the statement at `idx`, we are done.
                if from == to {
                    return;
                }

                from.statement_index + 1
            }
        };

        // Handle all statements between `from` and `to` whose effects must be applied in full.

        for statement_index in first_unapplied_index..to.statement_index {
            let location = Location { block, statement_index };
            let statement = &block_data.statements[statement_index];
            analysis.apply_before_statement_effect(state, statement, location);
            analysis.apply_statement_effect(state, statement, location);
        }

        // Handle the statement or terminator at `to`.

        let location = Location { block, statement_index: to.statement_index };
        if to.statement_index == terminator_index {
            let terminator = block_data.terminator();
            analysis.apply_before_terminator_effect(state, terminator, location);

            if to.effect == Effect::Primary {
                analysis.apply_terminator_effect(state, terminator, location);
            }
        } else {
            let statement = &block_data.statements[to.statement_index];
            analysis.apply_before_statement_effect(state, statement, location);

            if to.effect == Effect::Primary {
                analysis.apply_statement_effect(state, statement, location);
            }
        }
    }

    fn visit_results_in_block<'mir, 'tcx, F, R>(
        state: &mut F,
        block: BasicBlock,
        block_data: &'mir mir::BasicBlockData<'tcx>,
        results: &mut R,
        vis: &mut impl ResultsVisitor<'mir, 'tcx, R, FlowState = F>,
    ) where
        R: ResultsVisitable<'tcx, FlowState = F>,
    {
        results.reset_to_block_entry(state, block);

        vis.visit_block_start(results, state, block_data, block);

        for (statement_index, stmt) in block_data.statements.iter().enumerate() {
            let loc = Location { block, statement_index };
            results.reconstruct_before_statement_effect(state, stmt, loc);
            vis.visit_statement_before_primary_effect(results, state, stmt, loc);
            results.reconstruct_statement_effect(state, stmt, loc);
            vis.visit_statement_after_primary_effect(results, state, stmt, loc);
        }

        let loc = Location { block, statement_index: block_data.statements.len() };
        let term = block_data.terminator();
        results.reconstruct_before_terminator_effect(state, term, loc);
        vis.visit_terminator_before_primary_effect(results, state, term, loc);
        results.reconstruct_terminator_effect(state, term, loc);
        vis.visit_terminator_after_primary_effect(results, state, term, loc);

        vis.visit_block_end(results, state, block_data, block);
    }

    fn join_state_into_successors_of<'tcx, A>(
        analysis: &mut A,
        _tcx: TyCtxt<'tcx>,
        _body: &mir::Body<'tcx>,
        exit_state: &mut A::Domain,
        (bb, bb_data): (BasicBlock, &'_ mir::BasicBlockData<'tcx>),
        mut propagate: impl FnMut(BasicBlock, &A::Domain),
    ) where
        A: Analysis<'tcx>,
    {
        use mir::TerminatorKind::*;
        match bb_data.terminator().kind {
            Return | Resume | Terminate | GeneratorDrop | Unreachable => {}

            Goto { target } => propagate(target, exit_state),

            Assert { target, unwind, expected: _, msg: _, cond: _ }
            | Drop { target, unwind, place: _, replace: _ }
            | FalseUnwind { real_target: target, unwind } => {
                if let UnwindAction::Cleanup(unwind) = unwind {
                    propagate(unwind, exit_state);
                }

                propagate(target, exit_state);
            }

            FalseEdge { real_target, imaginary_target } => {
                propagate(real_target, exit_state);
                propagate(imaginary_target, exit_state);
            }

            Yield { resume: target, drop, resume_arg, value: _ } => {
                if let Some(drop) = drop {
                    propagate(drop, exit_state);
                }

                analysis.apply_yield_resume_effect(exit_state, target, resume_arg);
                propagate(target, exit_state);
            }

            Call { unwind, destination, target, func: _, args: _, call_source: _, fn_span: _ } => {
                if let UnwindAction::Cleanup(unwind) = unwind {
                    propagate(unwind, exit_state);
                }

                if let Some(target) = target {
                    // N.B.: This must be done *last*, otherwise the unwind path will see the call
                    // return effect.
                    analysis.apply_call_return_effect(
                        exit_state,
                        bb,
                        CallReturnPlaces::Call(destination),
                    );
                    propagate(target, exit_state);
                }
            }

            InlineAsm {
                template: _,
                ref operands,
                options: _,
                line_spans: _,
                destination,
                unwind,
            } => {
                if let UnwindAction::Cleanup(unwind) = unwind {
                    propagate(unwind, exit_state);
                }

                if let Some(target) = destination {
                    // N.B.: This must be done *last*, otherwise the unwind path will see the call
                    // return effect.
                    analysis.apply_call_return_effect(
                        exit_state,
                        bb,
                        CallReturnPlaces::InlineAsm(operands),
                    );
                    propagate(target, exit_state);
                }
            }

            SwitchInt { ref targets, ref discr } => {
                let mut applier = ForwardSwitchIntEdgeEffectsApplier {
                    exit_state,
                    targets,
                    propagate,
                    effects_applied: false,
                };

                analysis.apply_switch_int_edge_effects(bb, discr, &mut applier);

                let ForwardSwitchIntEdgeEffectsApplier {
                    exit_state,
                    mut propagate,
                    effects_applied,
                    ..
                } = applier;

                if !effects_applied {
                    for target in targets.all_targets() {
                        propagate(*target, exit_state);
                    }
                }
            }
        }
    }
}

struct ForwardSwitchIntEdgeEffectsApplier<'a, D, F> {
    exit_state: &'a mut D,
    targets: &'a SwitchTargets,
    propagate: F,

    effects_applied: bool,
}

impl<D, F> super::SwitchIntEdgeEffects<D> for ForwardSwitchIntEdgeEffectsApplier<'_, D, F>
where
    D: Clone,
    F: FnMut(BasicBlock, &D),
{
    fn apply(&mut self, mut apply_edge_effect: impl FnMut(&mut D, SwitchIntTarget)) {
        assert!(!self.effects_applied);

        let mut tmp = None;
        for (value, target) in self.targets.iter() {
            let tmp = opt_clone_from_or_clone(&mut tmp, self.exit_state);
            apply_edge_effect(tmp, SwitchIntTarget { value: Some(value), target });
            (self.propagate)(target, tmp);
        }

        // Once we get to the final, "otherwise" branch, there is no need to preserve `exit_state`,
        // so pass it directly to `apply_edge_effect` to save a clone of the dataflow state.
        let otherwise = self.targets.otherwise();
        apply_edge_effect(self.exit_state, SwitchIntTarget { value: None, target: otherwise });
        (self.propagate)(otherwise, self.exit_state);

        self.effects_applied = true;
    }
}

/// An analogue of `Option::get_or_insert_with` that stores a clone of `val` into `opt`, but uses
/// the more efficient `clone_from` if `opt` was `Some`.
///
/// Returns a mutable reference to the new clone that resides in `opt`.
//
// FIXME: Figure out how to express this using `Option::clone_from`, or maybe lift it into the
// standard library?
fn opt_clone_from_or_clone<'a, T: Clone>(opt: &'a mut Option<T>, val: &T) -> &'a mut T {
    if opt.is_some() {
        let ret = opt.as_mut().unwrap();
        ret.clone_from(val);
        ret
    } else {
        *opt = Some(val.clone());
        opt.as_mut().unwrap()
    }
}
