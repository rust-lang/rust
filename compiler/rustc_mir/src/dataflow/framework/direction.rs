use rustc_index::bit_set::BitSet;
use rustc_middle::mir::{self, BasicBlock, Location};
use rustc_middle::ty::{self, TyCtxt};
use std::ops::RangeInclusive;

use super::visitor::{ResultsVisitable, ResultsVisitor};
use super::{Analysis, Effect, EffectIndex, GenKillAnalysis, GenKillSet};

pub trait Direction {
    fn is_forward() -> bool;

    fn is_backward() -> bool {
        !Self::is_forward()
    }

    /// Applies all effects between the given `EffectIndex`s.
    ///
    /// `effects.start()` must precede or equal `effects.end()` in this direction.
    fn apply_effects_in_range<A>(
        analysis: &A,
        state: &mut BitSet<A::Idx>,
        block: BasicBlock,
        block_data: &mir::BasicBlockData<'tcx>,
        effects: RangeInclusive<EffectIndex>,
    ) where
        A: Analysis<'tcx>;

    fn apply_effects_in_block<A>(
        analysis: &A,
        state: &mut BitSet<A::Idx>,
        block: BasicBlock,
        block_data: &mir::BasicBlockData<'tcx>,
    ) where
        A: Analysis<'tcx>;

    fn gen_kill_effects_in_block<A>(
        analysis: &A,
        trans: &mut GenKillSet<A::Idx>,
        block: BasicBlock,
        block_data: &mir::BasicBlockData<'tcx>,
    ) where
        A: GenKillAnalysis<'tcx>;

    fn visit_results_in_block<F, R>(
        state: &mut F,
        block: BasicBlock,
        block_data: &'mir mir::BasicBlockData<'tcx>,
        results: &R,
        vis: &mut impl ResultsVisitor<'mir, 'tcx, FlowState = F>,
    ) where
        R: ResultsVisitable<'tcx, FlowState = F>;

    fn join_state_into_successors_of<A>(
        analysis: &A,
        tcx: TyCtxt<'tcx>,
        body: &mir::Body<'tcx>,
        dead_unwinds: Option<&BitSet<BasicBlock>>,
        exit_state: &mut BitSet<A::Idx>,
        block: (BasicBlock, &'_ mir::BasicBlockData<'tcx>),
        propagate: impl FnMut(BasicBlock, &BitSet<A::Idx>),
    ) where
        A: Analysis<'tcx>;
}

/// Dataflow that runs from the exit of a block (the terminator), to its entry (the first statement).
pub struct Backward;

impl Direction for Backward {
    fn is_forward() -> bool {
        false
    }

    fn apply_effects_in_block<A>(
        analysis: &A,
        state: &mut BitSet<A::Idx>,
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

    fn gen_kill_effects_in_block<A>(
        analysis: &A,
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

    fn apply_effects_in_range<A>(
        analysis: &A,
        state: &mut BitSet<A::Idx>,
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

    fn visit_results_in_block<F, R>(
        state: &mut F,
        block: BasicBlock,
        block_data: &'mir mir::BasicBlockData<'tcx>,
        results: &R,
        vis: &mut impl ResultsVisitor<'mir, 'tcx, FlowState = F>,
    ) where
        R: ResultsVisitable<'tcx, FlowState = F>,
    {
        results.reset_to_block_entry(state, block);

        vis.visit_block_end(&state, block_data, block);

        // Terminator
        let loc = Location { block, statement_index: block_data.statements.len() };
        let term = block_data.terminator();
        results.reconstruct_before_terminator_effect(state, term, loc);
        vis.visit_terminator_before_primary_effect(state, term, loc);
        results.reconstruct_terminator_effect(state, term, loc);
        vis.visit_terminator_after_primary_effect(state, term, loc);

        for (statement_index, stmt) in block_data.statements.iter().enumerate().rev() {
            let loc = Location { block, statement_index };
            results.reconstruct_before_statement_effect(state, stmt, loc);
            vis.visit_statement_before_primary_effect(state, stmt, loc);
            results.reconstruct_statement_effect(state, stmt, loc);
            vis.visit_statement_after_primary_effect(state, stmt, loc);
        }

        vis.visit_block_start(state, block_data, block);
    }

    fn join_state_into_successors_of<A>(
        analysis: &A,
        _tcx: TyCtxt<'tcx>,
        body: &mir::Body<'tcx>,
        dead_unwinds: Option<&BitSet<BasicBlock>>,
        exit_state: &mut BitSet<A::Idx>,
        (bb, _bb_data): (BasicBlock, &'_ mir::BasicBlockData<'tcx>),
        mut propagate: impl FnMut(BasicBlock, &BitSet<A::Idx>),
    ) where
        A: Analysis<'tcx>,
    {
        for pred in body.predecessors()[bb].iter().copied() {
            match body[pred].terminator().kind {
                // Apply terminator-specific edge effects.
                //
                // FIXME(ecstaticmorse): Avoid cloning the exit state unconditionally.
                mir::TerminatorKind::Call {
                    destination: Some((return_place, dest)),
                    ref func,
                    ref args,
                    ..
                } if dest == bb => {
                    let mut tmp = exit_state.clone();
                    analysis.apply_call_return_effect(&mut tmp, pred, func, args, return_place);
                    propagate(pred, &tmp);
                }

                mir::TerminatorKind::Yield { resume, resume_arg, .. } if resume == bb => {
                    let mut tmp = exit_state.clone();
                    analysis.apply_yield_resume_effect(&mut tmp, resume, resume_arg);
                    propagate(pred, &tmp);
                }

                // Ignore dead unwinds.
                mir::TerminatorKind::Call { cleanup: Some(unwind), .. }
                | mir::TerminatorKind::Assert { cleanup: Some(unwind), .. }
                | mir::TerminatorKind::Drop { unwind: Some(unwind), .. }
                | mir::TerminatorKind::DropAndReplace { unwind: Some(unwind), .. }
                | mir::TerminatorKind::FalseUnwind { unwind: Some(unwind), .. }
                    if unwind == bb =>
                {
                    if dead_unwinds.map_or(true, |dead| !dead.contains(bb)) {
                        propagate(pred, exit_state);
                    }
                }

                _ => propagate(pred, exit_state),
            }
        }
    }
}

/// Dataflow that runs from the entry of a block (the first statement), to its exit (terminator).
pub struct Forward;

impl Direction for Forward {
    fn is_forward() -> bool {
        true
    }

    fn apply_effects_in_block<A>(
        analysis: &A,
        state: &mut BitSet<A::Idx>,
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

    fn gen_kill_effects_in_block<A>(
        analysis: &A,
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

    fn apply_effects_in_range<A>(
        analysis: &A,
        state: &mut BitSet<A::Idx>,
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

    fn visit_results_in_block<F, R>(
        state: &mut F,
        block: BasicBlock,
        block_data: &'mir mir::BasicBlockData<'tcx>,
        results: &R,
        vis: &mut impl ResultsVisitor<'mir, 'tcx, FlowState = F>,
    ) where
        R: ResultsVisitable<'tcx, FlowState = F>,
    {
        results.reset_to_block_entry(state, block);

        vis.visit_block_start(state, block_data, block);

        for (statement_index, stmt) in block_data.statements.iter().enumerate() {
            let loc = Location { block, statement_index };
            results.reconstruct_before_statement_effect(state, stmt, loc);
            vis.visit_statement_before_primary_effect(state, stmt, loc);
            results.reconstruct_statement_effect(state, stmt, loc);
            vis.visit_statement_after_primary_effect(state, stmt, loc);
        }

        let loc = Location { block, statement_index: block_data.statements.len() };
        let term = block_data.terminator();
        results.reconstruct_before_terminator_effect(state, term, loc);
        vis.visit_terminator_before_primary_effect(state, term, loc);
        results.reconstruct_terminator_effect(state, term, loc);
        vis.visit_terminator_after_primary_effect(state, term, loc);

        vis.visit_block_end(state, block_data, block);
    }

    fn join_state_into_successors_of<A>(
        analysis: &A,
        tcx: TyCtxt<'tcx>,
        body: &mir::Body<'tcx>,
        dead_unwinds: Option<&BitSet<BasicBlock>>,
        exit_state: &mut BitSet<A::Idx>,
        (bb, bb_data): (BasicBlock, &'_ mir::BasicBlockData<'tcx>),
        mut propagate: impl FnMut(BasicBlock, &BitSet<A::Idx>),
    ) where
        A: Analysis<'tcx>,
    {
        use mir::TerminatorKind::*;
        match bb_data.terminator().kind {
            Return | Resume | Abort | GeneratorDrop | Unreachable => {}

            Goto { target } => propagate(target, exit_state),

            Assert { target, cleanup: unwind, expected: _, msg: _, cond: _ }
            | Drop { target, unwind, place: _ }
            | DropAndReplace { target, unwind, value: _, place: _ }
            | FalseUnwind { real_target: target, unwind } => {
                if let Some(unwind) = unwind {
                    if dead_unwinds.map_or(true, |dead| !dead.contains(bb)) {
                        propagate(unwind, exit_state);
                    }
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

            Call { cleanup, destination, ref func, ref args, from_hir_call: _, fn_span: _ } => {
                if let Some(unwind) = cleanup {
                    if dead_unwinds.map_or(true, |dead| !dead.contains(bb)) {
                        propagate(unwind, exit_state);
                    }
                }

                if let Some((dest_place, target)) = destination {
                    // N.B.: This must be done *last*, otherwise the unwind path will see the call
                    // return effect.
                    analysis.apply_call_return_effect(exit_state, bb, func, args, dest_place);
                    propagate(target, exit_state);
                }
            }

            InlineAsm { template: _, operands: _, options: _, line_spans: _, destination } => {
                if let Some(target) = destination {
                    propagate(target, exit_state);
                }
            }

            SwitchInt { ref targets, ref values, ref discr, switch_ty: _ } => {
                let enum_ = discr
                    .place()
                    .and_then(|discr| switch_on_enum_discriminant(tcx, &body, bb_data, discr));
                match enum_ {
                    // If this is a switch on an enum discriminant, a custom effect may be applied
                    // along each outgoing edge.
                    Some((enum_place, enum_def)) => {
                        // MIR building adds discriminants to the `values` array in the same order as they
                        // are yielded by `AdtDef::discriminants`. We rely on this to match each
                        // discriminant in `values` to its corresponding variant in linear time.
                        let mut tmp = BitSet::new_empty(exit_state.domain_size());
                        let mut discriminants = enum_def.discriminants(tcx);
                        for (value, target) in values.iter().zip(targets.iter().copied()) {
                            let (variant_idx, _) =
                                discriminants.find(|&(_, discr)| discr.val == *value).expect(
                                    "Order of `AdtDef::discriminants` differed \
                                         from that of `SwitchInt::values`",
                                );

                            tmp.overwrite(exit_state);
                            analysis.apply_discriminant_switch_effect(
                                &mut tmp,
                                bb,
                                enum_place,
                                enum_def,
                                variant_idx,
                            );
                            propagate(target, &tmp);
                        }

                        // Move out of `tmp` so we don't accidentally use it below.
                        std::mem::drop(tmp);

                        // Propagate dataflow state along the "otherwise" edge.
                        let otherwise = targets.last().copied().unwrap();
                        propagate(otherwise, exit_state)
                    }

                    // Otherwise, it's just a normal `SwitchInt`, and every successor sees the same
                    // exit state.
                    None => {
                        for target in targets.iter().copied() {
                            propagate(target, exit_state);
                        }
                    }
                }
            }
        }
    }
}

/// Inspect a `SwitchInt`-terminated basic block to see if the condition of that `SwitchInt` is
/// an enum discriminant.
///
/// We expect such blocks to have a call to `discriminant` as their last statement like so:
///   _42 = discriminant(_1)
///   SwitchInt(_42, ..)
///
/// If the basic block matches this pattern, this function returns the place corresponding to the
/// enum (`_1` in the example above) as well as the `AdtDef` of that enum.
fn switch_on_enum_discriminant(
    tcx: TyCtxt<'tcx>,
    body: &'mir mir::Body<'tcx>,
    block: &'mir mir::BasicBlockData<'tcx>,
    switch_on: mir::Place<'tcx>,
) -> Option<(mir::Place<'tcx>, &'tcx ty::AdtDef)> {
    match block.statements.last().map(|stmt| &stmt.kind) {
        Some(mir::StatementKind::Assign(box (lhs, mir::Rvalue::Discriminant(discriminated))))
            if *lhs == switch_on =>
        {
            match &discriminated.ty(body, tcx).ty.kind() {
                ty::Adt(def, _) => Some((*discriminated, def)),

                // `Rvalue::Discriminant` is also used to get the active yield point for a
                // generator, but we do not need edge-specific effects in that case. This may
                // change in the future.
                ty::Generator(..) => None,

                t => bug!("`discriminant` called on unexpected type {:?}", t),
            }
        }

        _ => None,
    }
}
