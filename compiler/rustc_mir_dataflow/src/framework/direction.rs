use std::cmp::Ordering;
use std::ops::RangeInclusive;

use rustc_middle::bug;
use rustc_middle::mir::{self, BasicBlock, CallReturnPlaces, Location, TerminatorEdges};

use super::visitor::ResultsVisitor;
use super::{Analysis, Effect, EffectIndex, SwitchTargetIndex};

pub trait Direction {
    const IS_FORWARD: bool;
    const IS_BACKWARD: bool = !Self::IS_FORWARD;

    /// Returns the first statement index for this direction. (0 when going forward and
    /// `statements.len()` when going backward.)
    fn first_index(block_data: &mir::BasicBlockData<'_>) -> EffectIndex;

    /// Returns `true` if `a` comes before `b` for this direction.
    fn index_precedes(a: EffectIndex, b: EffectIndex) -> bool;

    /// Returns the next index for this direction.
    fn next_index(idx: EffectIndex) -> EffectIndex;

    /// Called by `iterate_to_fixpoint` during initial analysis computation.
    fn apply_effects_in_block<'mir, 'tcx, A>(
        analysis: &A,
        body: &mir::Body<'tcx>,
        state: &mut A::Domain,
        block: BasicBlock,
        block_data: &'mir mir::BasicBlockData<'tcx>,
        propagate: impl FnMut(BasicBlock, &A::Domain),
    ) where
        A: Analysis<'tcx>;

    /// Called by `ResultsCursor` to recompute the domain value for a location
    /// in a basic block. Applies all effects between the given `EffectIndex`s.
    ///
    /// `effects.start()` must precede or equal `effects.end()` in this direction.
    fn apply_effects_in_range<'tcx, A>(
        analysis: &A,
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
        assert!(from.statement_index <= terminator_index);
        assert!(!Self::index_precedes(to, from));

        let mut idx = from;
        loop {
            analysis.apply_effect(state, block, block_data, idx);
            if idx == to {
                break;
            }
            idx = Self::next_index(idx);
        }
    }

    /// Called by `ResultsVisitor` to recompute the analysis domain values for
    /// all locations in a basic block (starting from `entry_state` and to
    /// visit them with `vis`.
    fn visit_results_in_block<'mir, 'tcx, A>(
        analysis: &A,
        state: &mut A::Domain,
        block: BasicBlock,
        block_data: &'mir mir::BasicBlockData<'tcx>,
        vis: &mut impl ResultsVisitor<'tcx, A>,
    ) where
        A: Analysis<'tcx>;
}

/// Dataflow that runs from the exit of a block (terminator), to its entry (the first statement).
pub struct Backward;

impl Direction for Backward {
    const IS_FORWARD: bool = false;

    fn first_index(block_data: &mir::BasicBlockData<'_>) -> EffectIndex {
        Effect::Early.at_index(block_data.statements.len())
    }

    fn index_precedes(a: EffectIndex, b: EffectIndex) -> bool {
        // Higher statement indices precede lower statement indices, and then `Early` effects
        // precede `Primary` effects. (That's why the two comparisons use different orders for `a`
        // and `b`.)
        let ord = b.statement_index.cmp(&a.statement_index).then_with(|| a.effect.cmp(&b.effect));
        ord == Ordering::Less
    }

    /// Returns the next index for this direction.
    fn next_index(idx: EffectIndex) -> EffectIndex {
        match idx.effect {
            Effect::Early => Effect::Primary.at_index(idx.statement_index),
            Effect::Primary => Effect::Early.at_index(idx.statement_index - 1),
        }
    }

    fn apply_effects_in_block<'mir, 'tcx, A>(
        analysis: &A,
        body: &mir::Body<'tcx>,
        state: &mut A::Domain,
        block: BasicBlock,
        block_data: &'mir mir::BasicBlockData<'tcx>,
        mut propagate: impl FnMut(BasicBlock, &A::Domain),
    ) where
        A: Analysis<'tcx>,
    {
        let terminator = block_data.terminator();
        let location = Location { block, statement_index: block_data.statements.len() };
        analysis.apply_early_terminator_effect(state, terminator, location);
        analysis.apply_primary_terminator_effect(state, terminator, location);
        for (statement_index, statement) in block_data.statements.iter().enumerate().rev() {
            let location = Location { block, statement_index };
            analysis.apply_early_statement_effect(state, statement, location);
            analysis.apply_primary_statement_effect(state, statement, location);
        }

        let exit_state = state;
        for pred in body.basic_blocks.predecessors()[block].iter().copied() {
            match body[pred].terminator().kind {
                // Apply terminator-specific edge effects.
                mir::TerminatorKind::Call { destination, target: Some(dest), .. }
                    if dest == block =>
                {
                    let mut tmp = exit_state.clone();
                    analysis.apply_call_return_effect(
                        &mut tmp,
                        pred,
                        CallReturnPlaces::Call(destination),
                    );
                    propagate(pred, &tmp);
                }

                mir::TerminatorKind::InlineAsm { ref targets, ref operands, .. }
                    if targets.contains(&block) =>
                {
                    let mut tmp = exit_state.clone();
                    analysis.apply_call_return_effect(
                        &mut tmp,
                        pred,
                        CallReturnPlaces::InlineAsm(operands),
                    );
                    propagate(pred, &tmp);
                }

                mir::TerminatorKind::Yield { resume, drop, resume_arg, .. }
                    if resume == block || drop == Some(block) =>
                {
                    let mut tmp = exit_state.clone();
                    analysis.apply_call_return_effect(
                        &mut tmp,
                        block,
                        CallReturnPlaces::Yield(resume_arg),
                    );
                    propagate(pred, &tmp);
                }

                mir::TerminatorKind::SwitchInt { ref targets, ref discr } => {
                    if let Some(_data) = analysis.get_switch_int_data(pred, targets, discr) {
                        bug!(
                            "SwitchInt edge effects are unsupported in backward dataflow analyses"
                        );
                    } else {
                        propagate(pred, exit_state)
                    }
                }

                _ => propagate(pred, exit_state),
            }
        }
    }

    fn visit_results_in_block<'mir, 'tcx, A>(
        analysis: &A,
        state: &mut A::Domain,
        block: BasicBlock,
        block_data: &'mir mir::BasicBlockData<'tcx>,
        vis: &mut impl ResultsVisitor<'tcx, A>,
    ) where
        A: Analysis<'tcx>,
    {
        vis.visit_block_entry(state);

        let loc = Location { block, statement_index: block_data.statements.len() };
        let term = block_data.terminator();
        analysis.apply_early_terminator_effect(state, term, loc);
        vis.visit_after_early_terminator_effect(analysis, state, term, loc);
        analysis.apply_primary_terminator_effect(state, term, loc);
        vis.visit_after_primary_terminator_effect(analysis, state, term, loc);

        for (statement_index, stmt) in block_data.statements.iter().enumerate().rev() {
            let loc = Location { block, statement_index };
            analysis.apply_early_statement_effect(state, stmt, loc);
            vis.visit_after_early_statement_effect(analysis, state, stmt, loc);
            analysis.apply_primary_statement_effect(state, stmt, loc);
            vis.visit_after_primary_statement_effect(analysis, state, stmt, loc);
        }
    }
}

/// Dataflow that runs from the entry of a block (the first statement), to its exit (terminator).
pub struct Forward;

impl Direction for Forward {
    const IS_FORWARD: bool = true;

    fn first_index(_block_data: &mir::BasicBlockData<'_>) -> EffectIndex {
        Effect::Early.at_index(0)
    }

    fn index_precedes(a: EffectIndex, b: EffectIndex) -> bool {
        // Lower statement indices precede higher statement indices, and then `Early` effects
        // precede `Primary` effects.
        let ord = a.statement_index.cmp(&b.statement_index).then_with(|| a.effect.cmp(&b.effect));
        ord == Ordering::Less
    }

    /// Returns the next index for this direction.
    fn next_index(idx: EffectIndex) -> EffectIndex {
        match idx.effect {
            Effect::Early => Effect::Primary.at_index(idx.statement_index),
            Effect::Primary => Effect::Early.at_index(idx.statement_index + 1),
        }
    }

    fn apply_effects_in_block<'mir, 'tcx, A>(
        analysis: &A,
        body: &mir::Body<'tcx>,
        state: &mut A::Domain,
        block: BasicBlock,
        block_data: &'mir mir::BasicBlockData<'tcx>,
        mut propagate: impl FnMut(BasicBlock, &A::Domain),
    ) where
        A: Analysis<'tcx>,
    {
        for (statement_index, statement) in block_data.statements.iter().enumerate() {
            let location = Location { block, statement_index };
            analysis.apply_early_statement_effect(state, statement, location);
            analysis.apply_primary_statement_effect(state, statement, location);
        }
        let terminator = block_data.terminator();
        let location = Location { block, statement_index: block_data.statements.len() };
        analysis.apply_early_terminator_effect(state, terminator, location);
        let edges = analysis.apply_primary_terminator_effect(state, terminator, location);

        let exit_state = state;
        match edges {
            TerminatorEdges::None => {}
            TerminatorEdges::Single(target) => propagate(target, exit_state),
            TerminatorEdges::Double(target, unwind) => {
                propagate(target, exit_state);
                propagate(unwind, exit_state);
            }
            TerminatorEdges::AssignOnReturn { return_, cleanup, place } => {
                // This must be done *first*, otherwise the unwind path will see the assignments.
                if let Some(cleanup) = cleanup {
                    propagate(cleanup, exit_state);
                }

                if !return_.is_empty() {
                    analysis.apply_call_return_effect(exit_state, block, place);
                    for target in return_ {
                        propagate(target, exit_state);
                    }
                }
            }
            TerminatorEdges::SwitchInt { targets, discr } => {
                if let Some(mut data) = analysis.get_switch_int_data(block, targets, discr) {
                    let mut tmp = analysis.bottom_value(body);
                    for (i, (_value, target)) in targets.iter().enumerate() {
                        tmp.clone_from(exit_state);
                        let target_idx = SwitchTargetIndex::Normal(i);
                        analysis.apply_switch_int_edge_effect(&mut tmp, &mut data, target_idx);
                        propagate(target, &tmp);
                    }

                    // Once we get to the final, "otherwise" branch, there is no need to preserve
                    // `exit_state`, so pass it directly to `apply_switch_int_edge_effect` to save
                    // a clone of the dataflow state.
                    analysis.apply_switch_int_edge_effect(
                        exit_state,
                        &mut data,
                        SwitchTargetIndex::Otherwise,
                    );
                    propagate(targets.otherwise(), exit_state);
                } else {
                    for target in targets.all_targets() {
                        propagate(*target, exit_state);
                    }
                }
            }
        }
    }

    fn visit_results_in_block<'mir, 'tcx, A>(
        analysis: &A,
        state: &mut A::Domain,
        block: BasicBlock,
        block_data: &'mir mir::BasicBlockData<'tcx>,
        vis: &mut impl ResultsVisitor<'tcx, A>,
    ) where
        A: Analysis<'tcx>,
    {
        vis.visit_block_entry(state);

        for (statement_index, stmt) in block_data.statements.iter().enumerate() {
            let loc = Location { block, statement_index };
            analysis.apply_early_statement_effect(state, stmt, loc);
            vis.visit_after_early_statement_effect(analysis, state, stmt, loc);
            analysis.apply_primary_statement_effect(state, stmt, loc);
            vis.visit_after_primary_statement_effect(analysis, state, stmt, loc);
        }

        let loc = Location { block, statement_index: block_data.statements.len() };
        let term = block_data.terminator();
        analysis.apply_early_terminator_effect(state, term, loc);
        vis.visit_after_early_terminator_effect(analysis, state, term, loc);
        analysis.apply_primary_terminator_effect(state, term, loc);
        vis.visit_after_primary_terminator_effect(analysis, state, term, loc);
    }
}
