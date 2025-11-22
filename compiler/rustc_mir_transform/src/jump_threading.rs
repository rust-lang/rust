//! A jump threading optimization.
//!
//! This optimization seeks to replace join-then-switch control flow patterns by straight jumps
//!    X = 0                                      X = 0
//! ------------\      /--------              ------------
//!    X = 1     X----X SwitchInt(X)     =>       X = 1
//! ------------/      \--------              ------------
//!
//!
//! We proceed by walking the cfg backwards starting from each `SwitchInt` terminator,
//! looking for assignments that will turn the `SwitchInt` into a simple `Goto`.
//!
//! The algorithm maintains a set of replacement conditions:
//! - `conditions[place]` contains `Condition { value, polarity: Eq, target }`
//!   if assigning `value` to `place` turns the `SwitchInt` into `Goto { target }`.
//! - `conditions[place]` contains `Condition { value, polarity: Ne, target }`
//!   if assigning anything different from `value` to `place` turns the `SwitchInt`
//!   into `Goto { target }`.
//!
//! In this file, we denote as `place ?= value` the existence of a replacement condition
//! on `place` with given `value`, irrespective of the polarity and target of that
//! replacement condition.
//!
//! We then walk the CFG backwards transforming the set of conditions.
//! When we find a fulfilling assignment, we record a `ThreadingOpportunity`.
//! All `ThreadingOpportunity`s are applied to the body, by duplicating blocks if required.
//!
//! The optimization search can be very heavy, as it performs a DFS on MIR starting from
//! each `SwitchInt` terminator. To manage the complexity, we:
//! - bound the maximum depth by a constant `MAX_BACKTRACK`;
//! - we only traverse `Goto` terminators.
//!
//! We try to avoid creating irreducible control-flow by not threading through a loop header.
//!
//! Likewise, applying the optimisation can create a lot of new MIR, so we bound the instruction
//! cost by `MAX_COST`.

use rustc_arena::DroplessArena;
use rustc_const_eval::const_eval::DummyMachine;
use rustc_const_eval::interpret::{ImmTy, Immediate, InterpCx, OpTy, Projectable};
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_index::IndexVec;
use rustc_index::bit_set::DenseBitSet;
use rustc_middle::bug;
use rustc_middle::mir::interpret::Scalar;
use rustc_middle::mir::visit::Visitor;
use rustc_middle::mir::*;
use rustc_middle::ty::{self, ScalarInt, TyCtxt};
use rustc_mir_dataflow::lattice::HasBottom;
use rustc_mir_dataflow::value_analysis::{
    Map, PlaceCollectionMode, PlaceIndex, TrackElem, ValueIndex,
};
use rustc_span::DUMMY_SP;
use tracing::{debug, instrument, trace};

use crate::cost_checker::CostChecker;

pub(super) struct JumpThreading;

const MAX_BACKTRACK: usize = 5;
const MAX_COST: usize = 100;

impl<'tcx> crate::MirPass<'tcx> for JumpThreading {
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

        let typing_env = body.typing_env(tcx);
        let arena = &DroplessArena::default();
        let mut finder = TOFinder {
            tcx,
            typing_env,
            ecx: InterpCx::new(tcx, DUMMY_SP, typing_env, DummyMachine),
            body,
            arena,
            map: Map::new(tcx, body, PlaceCollectionMode::OnDemand),
            maybe_loop_headers: loops::maybe_loop_headers(body),
            opportunities: Vec::new(),
        };

        for (bb, _) in traversal::preorder(body) {
            finder.start_from_switch(bb);
        }

        let opportunities = finder.opportunities;
        debug!(?opportunities);
        if opportunities.is_empty() {
            return;
        }

        // Verify that we do not thread through a loop header.
        for to in opportunities.iter() {
            assert!(to.chain.iter().all(|&block| !finder.maybe_loop_headers.contains(block)));
        }
        OpportunitySet::new(body, opportunities).apply(body);
    }

    fn is_required(&self) -> bool {
        false
    }
}

#[derive(Debug)]
struct ThreadingOpportunity {
    /// The list of `BasicBlock`s from the one that found the opportunity to the `SwitchInt`.
    chain: Vec<BasicBlock>,
    /// The `SwitchInt` will be replaced by `Goto { target }`.
    target: BasicBlock,
}

struct TOFinder<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    typing_env: ty::TypingEnv<'tcx>,
    ecx: InterpCx<'tcx, DummyMachine>,
    body: &'a Body<'tcx>,
    map: Map<'tcx>,
    maybe_loop_headers: DenseBitSet<BasicBlock>,
    /// We use an arena to avoid cloning the slices when cloning `state`.
    arena: &'a DroplessArena,
    opportunities: Vec<ThreadingOpportunity>,
}

/// Represent the following statement. If we can prove that the current local is equal/not-equal
/// to `value`, jump to `target`.
#[derive(Copy, Clone, Debug)]
struct Condition {
    place: ValueIndex,
    value: ScalarInt,
    polarity: Polarity,
    target: BasicBlock,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum Polarity {
    Ne,
    Eq,
}

impl Condition {
    fn matches(&self, place: ValueIndex, value: ScalarInt) -> bool {
        self.place == place && (self.value == value) == (self.polarity == Polarity::Eq)
    }

    fn into_opportunity(self, match_bb: Option<BasicBlock>) -> ThreadingOpportunity {
        trace!(?self, "registering");
        let chain = match_bb.into_iter().collect();
        ThreadingOpportunity { chain, target: self.target }
    }
}

#[derive(Copy, Clone, Debug)]
struct ConditionSet<'a>(&'a [Condition]);

impl HasBottom for ConditionSet<'_> {
    const BOTTOM: Self = ConditionSet(&[]);

    fn is_bottom(&self) -> bool {
        self.0.is_empty()
    }
}

impl<'a> ConditionSet<'a> {
    fn is_empty(self) -> bool {
        self.0.is_empty()
    }

    fn iter(self) -> impl Iterator<Item = Condition> {
        self.0.iter().copied()
    }

    fn iter_matches(self, place: ValueIndex, value: ScalarInt) -> impl Iterator<Item = Condition> {
        self.iter().filter(move |c| c.matches(place, value))
    }

    fn register_matches(
        &self,
        place: ValueIndex,
        value: ScalarInt,
        match_bb: Option<BasicBlock>,
        opportunities: &mut Vec<ThreadingOpportunity>,
    ) {
        self.iter_matches(place, value)
            .for_each(|cond| opportunities.push(cond.into_opportunity(match_bb)))
    }

    fn filter(self, arena: &'a DroplessArena, f: impl Fn(Condition) -> bool) -> ConditionSet<'a> {
        let set = arena.alloc_from_iter(self.iter().filter(|&c| f(c)));
        ConditionSet(set)
    }

    fn filter_map(
        self,
        arena: &'a DroplessArena,
        f: impl Fn(Condition) -> Option<Condition>,
    ) -> ConditionSet<'a> {
        let set = arena.alloc_from_iter(self.iter().filter_map(|c| f(c)));
        ConditionSet(set)
    }

    fn map(self, arena: &'a DroplessArena, f: impl Fn(Condition) -> Condition) -> ConditionSet<'a> {
        let set = arena.alloc_from_iter(self.iter().map(|c| f(c)));
        ConditionSet(set)
    }
}

impl<'a, 'tcx> TOFinder<'a, 'tcx> {
    fn place(&mut self, place: Place<'tcx>, tail: Option<TrackElem>) -> Option<PlaceIndex> {
        self.map.register_place(self.tcx, self.body, place, tail)
    }

    fn value(&mut self, place: PlaceIndex) -> Option<ValueIndex> {
        self.map.register_value(self.tcx, self.typing_env, place)
    }

    fn place_value(&mut self, place: Place<'tcx>, tail: Option<TrackElem>) -> Option<ValueIndex> {
        let place = self.place(place, tail)?;
        self.value(place)
    }

    /// Recursion entry point to find threading opportunities.
    #[instrument(level = "trace", skip(self))]
    fn start_from_switch(&mut self, bb: BasicBlock) {
        let bbdata = &self.body[bb];
        if bbdata.is_cleanup || self.maybe_loop_headers.contains(bb) {
            return;
        }
        let Some((discr, targets)) = bbdata.terminator().kind.as_switch() else { return };
        let Some(discr) = discr.place() else { return };
        debug!(?discr, ?bb);

        let discr_ty = discr.ty(self.body, self.tcx).ty;
        let Ok(discr_layout) = self.ecx.layout_of(discr_ty) else { return };

        let Some(discr) = self.place_value(discr, None) else { return };
        debug!(?discr);

        let cost = CostChecker::new(self.tcx, self.typing_env, None, self.body);

        let conds = if let Some((value, then, else_)) = targets.as_static_if() {
            let Some(value) = ScalarInt::try_from_uint(value, discr_layout.size) else { return };
            self.arena.alloc_from_iter([
                Condition { place: discr, value, polarity: Polarity::Eq, target: then },
                Condition { place: discr, value, polarity: Polarity::Ne, target: else_ },
            ])
        } else {
            self.arena.alloc_from_iter(targets.iter().filter_map(|(value, target)| {
                let value = ScalarInt::try_from_uint(value, discr_layout.size)?;
                Some(Condition { place: discr, value, polarity: Polarity::Eq, target })
            }))
        };
        let state = ConditionSet(conds);
        self.find_opportunity(bb, state, cost, 0)
    }

    /// Recursively walk statements backwards from this bb's terminator to find threading
    /// opportunities.
    #[instrument(level = "trace", skip(self, cost), ret)]
    fn find_opportunity(
        &mut self,
        bb: BasicBlock,
        mut state: ConditionSet<'a>,
        mut cost: CostChecker<'_, 'tcx>,
        depth: usize,
    ) {
        // Do not thread through loop headers.
        if self.maybe_loop_headers.contains(bb) {
            return;
        }

        debug!(cost = ?cost.cost());
        for (statement_index, stmt) in
            self.body.basic_blocks[bb].statements.iter().enumerate().rev()
        {
            if state.is_empty() {
                return;
            }

            cost.visit_statement(stmt, Location { block: bb, statement_index });
            if cost.cost() > MAX_COST {
                return;
            }

            // Attempt to turn the `current_condition` on `lhs` into a condition on another place.
            self.process_statement(bb, stmt, &mut state);

            // When a statement mutates a place, assignments to that place that happen
            // above the mutation cannot fulfill a condition.
            //   _1 = 5 // Whatever happens here, it won't change the result of a `SwitchInt`.
            //   _1 = 6
            if let Some((lhs, tail)) = self.mutated_statement(stmt) {
                self.flood_state(lhs, tail, &mut state);
            }
        }

        if state.is_empty() || depth >= MAX_BACKTRACK {
            return;
        }

        let last_non_rec = self.opportunities.len();

        let predecessors = &self.body.basic_blocks.predecessors()[bb];
        if let &[pred] = &predecessors[..]
            && bb != START_BLOCK
        {
            let term = self.body.basic_blocks[pred].terminator();
            match term.kind {
                TerminatorKind::SwitchInt { ref discr, ref targets } => {
                    self.process_switch_int(discr, targets, bb, &mut state);
                    self.find_opportunity(pred, state, cost, depth + 1);
                }
                _ => self.recurse_through_terminator(pred, state, &cost, depth),
            }
        } else if let &[ref predecessors @ .., last_pred] = &predecessors[..] {
            for &pred in predecessors {
                self.recurse_through_terminator(pred, state, &cost, depth);
            }
            self.recurse_through_terminator(last_pred, state, &cost, depth);
        }

        let new_tos = &mut self.opportunities[last_non_rec..];
        debug!(?new_tos);

        // Try to deduplicate threading opportunities.
        if new_tos.len() > 1
            && new_tos.len() == predecessors.len()
            && predecessors
                .iter()
                .zip(new_tos.iter())
                .all(|(&pred, to)| to.chain == &[pred] && to.target == new_tos[0].target)
        {
            // All predecessors have a threading opportunity, and they all point to the same block.
            debug!(?new_tos, "dedup");
            let first = &mut new_tos[0];
            *first = ThreadingOpportunity { chain: vec![bb], target: first.target };
            self.opportunities.truncate(last_non_rec + 1);
            return;
        }

        for op in self.opportunities[last_non_rec..].iter_mut() {
            op.chain.push(bb);
        }
    }

    /// Remove all conditions in the state that alias given place.
    fn flood_state(
        &self,
        place: Place<'tcx>,
        extra_elem: Option<TrackElem>,
        state: &mut ConditionSet<'a>,
    ) {
        let mut places_to_exclude = FxHashSet::default();
        self.map.for_each_aliasing_place(place.as_ref(), extra_elem, &mut |vi| {
            places_to_exclude.insert(vi);
        });
        if places_to_exclude.is_empty() {
            return;
        }
        *state = state.filter(self.arena, |c| !places_to_exclude.contains(&c.place));
    }

    /// Extract the mutated place from a statement.
    ///
    /// This method returns the `Place` so we can flood the state in case of a partial assignment.
    ///     (_1 as Ok).0 = _5;
    ///     (_1 as Err).0 = _6;
    /// We want to ensure that a `SwitchInt((_1 as Ok).0)` does not see the first assignment, as
    /// the value may have been mangled by the second assignment.
    ///
    /// In case we assign to a discriminant, we return `Some(TrackElem::Discriminant)`, so we can
    /// stop at flooding the discriminant, and preserve the variant fields.
    ///     (_1 as Some).0 = _6;
    ///     SetDiscriminant(_1, 1);
    ///     switchInt((_1 as Some).0)
    #[instrument(level = "trace", skip(self), ret)]
    fn mutated_statement(
        &self,
        stmt: &Statement<'tcx>,
    ) -> Option<(Place<'tcx>, Option<TrackElem>)> {
        match stmt.kind {
            StatementKind::Assign(box (place, _)) => Some((place, None)),
            StatementKind::SetDiscriminant { box place, variant_index: _ } => {
                Some((place, Some(TrackElem::Discriminant)))
            }
            StatementKind::StorageLive(local) | StatementKind::StorageDead(local) => {
                Some((Place::from(local), None))
            }
            StatementKind::Retag(..)
            | StatementKind::Intrinsic(box NonDivergingIntrinsic::Assume(..))
            // copy_nonoverlapping takes pointers and mutated the pointed-to value.
            | StatementKind::Intrinsic(box NonDivergingIntrinsic::CopyNonOverlapping(..))
            | StatementKind::AscribeUserType(..)
            | StatementKind::Coverage(..)
            | StatementKind::FakeRead(..)
            | StatementKind::ConstEvalCounter
            | StatementKind::PlaceMention(..)
            | StatementKind::BackwardIncompatibleDropHint { .. }
            | StatementKind::Nop => None,
        }
    }

    #[instrument(level = "trace", skip(self))]
    fn process_immediate(
        &mut self,
        bb: BasicBlock,
        lhs: PlaceIndex,
        rhs: ImmTy<'tcx>,
        state: &mut ConditionSet<'a>,
    ) {
        if let Some(lhs) = self.value(lhs)
            && let Immediate::Scalar(Scalar::Int(int)) = *rhs
        {
            state.register_matches(lhs, int, Some(bb), &mut self.opportunities);
        }
    }

    /// If we expect `lhs ?= A`, we have an opportunity if we assume `constant == A`.
    #[instrument(level = "trace", skip(self))]
    fn process_constant(
        &mut self,
        bb: BasicBlock,
        lhs: PlaceIndex,
        constant: OpTy<'tcx>,
        state: &mut ConditionSet<'a>,
    ) {
        self.map.for_each_projection_value(
            lhs,
            constant,
            &mut |elem, op| match elem {
                TrackElem::Field(idx) => self.ecx.project_field(op, idx).discard_err(),
                TrackElem::Variant(idx) => self.ecx.project_downcast(op, idx).discard_err(),
                TrackElem::Discriminant => {
                    let variant = self.ecx.read_discriminant(op).discard_err()?;
                    let discr_value =
                        self.ecx.discriminant_for_variant(op.layout.ty, variant).discard_err()?;
                    Some(discr_value.into())
                }
                TrackElem::DerefLen => {
                    let op: OpTy<'_> = self.ecx.deref_pointer(op).discard_err()?.into();
                    let len_usize = op.len(&self.ecx).discard_err()?;
                    let layout = self.ecx.layout_of(self.tcx.types.usize).unwrap();
                    Some(ImmTy::from_uint(len_usize, layout).into())
                }
            },
            &mut |place, op| {
                if let Some(place) = self.map.value(place)
                    && let Some(imm) = self.ecx.read_immediate_raw(op).discard_err()
                    && let Some(imm) = imm.right()
                    && let Immediate::Scalar(Scalar::Int(int)) = *imm
                {
                    state.register_matches(place, int, Some(bb), &mut self.opportunities);
                }
            },
        );
    }

    #[instrument(level = "trace", skip(self))]
    fn process_copy(&mut self, lhs: PlaceIndex, rhs: PlaceIndex, state: &mut ConditionSet<'a>) {
        let mut renames = FxHashMap::default();
        self.map.register_copy_tree(
            lhs, // tree to copy
            rhs, // tree to build
            &mut |lhs, rhs| {
                renames.insert(lhs, rhs);
            },
        );
        *state = state.map(self.arena, |mut c| {
            if let Some(rhs) = renames.get(&c.place) {
                c.place = *rhs
            }
            c
        });
    }

    #[instrument(level = "trace", skip(self))]
    fn process_operand(
        &mut self,
        bb: BasicBlock,
        lhs: PlaceIndex,
        rhs: &Operand<'tcx>,
        state: &mut ConditionSet<'a>,
    ) {
        match rhs {
            // If we expect `lhs ?= A`, we have an opportunity if we assume `constant == A`.
            Operand::Constant(constant) => {
                let Some(constant) =
                    self.ecx.eval_mir_constant(&constant.const_, constant.span, None).discard_err()
                else {
                    return;
                };
                self.process_constant(bb, lhs, constant, state);
            }
            // Transfer the conditions on the copied rhs.
            Operand::Move(rhs) | Operand::Copy(rhs) => {
                let Some(rhs) = self.place(*rhs, None) else { return };
                self.process_copy(lhs, rhs, state)
            }
        }
    }

    #[instrument(level = "trace", skip(self))]
    fn process_assign(
        &mut self,
        bb: BasicBlock,
        lhs_place: &Place<'tcx>,
        rvalue: &Rvalue<'tcx>,
        state: &mut ConditionSet<'a>,
    ) {
        let Some(lhs) = self.place(*lhs_place, None) else { return };
        match rvalue {
            Rvalue::Use(operand) => self.process_operand(bb, lhs, operand, state),
            // Transfer the conditions on the copy rhs.
            Rvalue::Discriminant(rhs) => {
                let Some(rhs) = self.place(*rhs, Some(TrackElem::Discriminant)) else { return };
                self.process_copy(lhs, rhs, state)
            }
            // If we expect `lhs ?= A`, we have an opportunity if we assume `constant == A`.
            Rvalue::Aggregate(box kind, operands) => {
                let agg_ty = lhs_place.ty(self.body, self.tcx).ty;
                let lhs = match kind {
                    // Do not support unions.
                    AggregateKind::Adt(.., Some(_)) => return,
                    AggregateKind::Adt(_, variant_index, ..) if agg_ty.is_enum() => {
                        let discr_ty = agg_ty.discriminant_ty(self.tcx);
                        let discr_target =
                            self.map.register_place_index(discr_ty, lhs, TrackElem::Discriminant);
                        if let Some(discr_value) =
                            self.ecx.discriminant_for_variant(agg_ty, *variant_index).discard_err()
                        {
                            self.process_immediate(bb, discr_target, discr_value, state);
                        }
                        self.map.register_place_index(
                            agg_ty,
                            lhs,
                            TrackElem::Variant(*variant_index),
                        )
                    }
                    _ => lhs,
                };
                for (field_index, operand) in operands.iter_enumerated() {
                    let operand_ty = operand.ty(self.body, self.tcx);
                    let field = self.map.register_place_index(
                        operand_ty,
                        lhs,
                        TrackElem::Field(field_index),
                    );
                    self.process_operand(bb, field, operand, state);
                }
            }
            // Transfer the conditions on the copy rhs, after inverting the value of the condition.
            Rvalue::UnaryOp(UnOp::Not, Operand::Move(operand) | Operand::Copy(operand)) => {
                let layout = self.ecx.layout_of(operand.ty(self.body, self.tcx).ty).unwrap();
                let Some(lhs) = self.value(lhs) else { return };
                let Some(operand) = self.place_value(*operand, None) else { return };
                *state = state.filter_map(self.arena, |mut cond| {
                    if cond.place == lhs {
                        cond.place = operand;
                        cond.value = self
                            .ecx
                            .unary_op(UnOp::Not, &ImmTy::from_scalar_int(cond.value, layout))
                            .discard_err()?
                            .to_scalar_int()
                            .discard_err()?;
                    }
                    Some(cond)
                });
            }
            // We expect `lhs ?= A`. We found `lhs = Eq(rhs, B)`.
            // Create a condition on `rhs ?= B`.
            Rvalue::BinaryOp(
                op,
                box (Operand::Move(operand) | Operand::Copy(operand), Operand::Constant(value))
                | box (Operand::Constant(value), Operand::Move(operand) | Operand::Copy(operand)),
            ) => {
                let equals = match op {
                    BinOp::Eq => ScalarInt::TRUE,
                    BinOp::Ne => ScalarInt::FALSE,
                    _ => return,
                };
                if value.const_.ty().is_floating_point() {
                    // Floating point equality does not follow bit-patterns.
                    // -0.0 and NaN both have special rules for equality,
                    // and therefore we cannot use integer comparisons for them.
                    // Avoid handling them, though this could be extended in the future.
                    return;
                }
                let Some(lhs) = self.value(lhs) else { return };
                let Some(operand) = self.place_value(*operand, None) else { return };
                let Some(value) = value.const_.try_eval_scalar_int(self.tcx, self.typing_env)
                else {
                    return;
                };
                *state = state.map(self.arena, |c| {
                    if c.place == lhs {
                        Condition {
                            place: operand,
                            value,
                            polarity: if c.matches(lhs, equals) {
                                Polarity::Eq
                            } else {
                                Polarity::Ne
                            },
                            ..c
                        }
                    } else {
                        c
                    }
                });
            }

            _ => {}
        }
    }

    #[instrument(level = "trace", skip(self))]
    fn process_statement(
        &mut self,
        bb: BasicBlock,
        stmt: &Statement<'tcx>,
        state: &mut ConditionSet<'a>,
    ) {
        // Below, `lhs` is the return value of `mutated_statement`,
        // the place to which `conditions` apply.

        match &stmt.kind {
            // If we expect `discriminant(place) ?= A`,
            // we have an opportunity if `variant_index ?= A`.
            StatementKind::SetDiscriminant { box place, variant_index } => {
                let Some(discr_target) = self.place(*place, Some(TrackElem::Discriminant)) else {
                    return;
                };
                let enum_ty = place.ty(self.body, self.tcx).ty;
                // `SetDiscriminant` guarantees that the discriminant is now `variant_index`.
                // Even if the discriminant write does nothing due to niches, it is UB to set the
                // discriminant when the data does not encode the desired discriminant.
                let Some(discr) =
                    self.ecx.discriminant_for_variant(enum_ty, *variant_index).discard_err()
                else {
                    return;
                };
                self.process_immediate(bb, discr_target, discr, state)
            }
            // If we expect `lhs ?= true`, we have an opportunity if we assume `lhs == true`.
            StatementKind::Intrinsic(box NonDivergingIntrinsic::Assume(
                Operand::Copy(place) | Operand::Move(place),
            )) => {
                let Some(place) = self.place_value(*place, None) else { return };
                state.register_matches(place, ScalarInt::TRUE, Some(bb), &mut self.opportunities)
            }
            StatementKind::Assign(box (lhs_place, rhs)) => {
                self.process_assign(bb, lhs_place, rhs, state)
            }
            _ => {}
        }
    }

    #[instrument(level = "trace", skip(self, state, cost))]
    fn recurse_through_terminator(
        &mut self,
        bb: BasicBlock,
        mut state: ConditionSet<'a>,
        cost: &CostChecker<'_, 'tcx>,
        depth: usize,
    ) {
        let term = self.body.basic_blocks[bb].terminator();
        let place_to_flood = match term.kind {
            // We come from a target, so those are not possible.
            TerminatorKind::UnwindResume
            | TerminatorKind::UnwindTerminate(_)
            | TerminatorKind::Return
            | TerminatorKind::TailCall { .. }
            | TerminatorKind::Unreachable
            | TerminatorKind::CoroutineDrop => bug!("{term:?} has no terminators"),
            // Disallowed during optimizations.
            TerminatorKind::FalseEdge { .. }
            | TerminatorKind::FalseUnwind { .. }
            | TerminatorKind::Yield { .. } => bug!("{term:?} invalid"),
            // Cannot reason about inline asm.
            TerminatorKind::InlineAsm { .. } => return,
            // `SwitchInt` is handled specially.
            TerminatorKind::SwitchInt { .. } => return,
            // We can recurse, no thing particular to do.
            TerminatorKind::Goto { .. } => None,
            // Flood the overwritten place, and progress through.
            TerminatorKind::Drop { place: destination, .. }
            | TerminatorKind::Call { destination, .. } => Some(destination),
            // Ignore, as this can be a no-op at codegen time.
            TerminatorKind::Assert { .. } => None,
        };

        // We can recurse through this terminator.
        if let Some(place_to_flood) = place_to_flood {
            self.flood_state(place_to_flood, None, &mut state);
        }
        self.find_opportunity(bb, state, cost.clone(), depth + 1)
    }

    #[instrument(level = "trace", skip(self))]
    fn process_switch_int(
        &mut self,
        discr: &Operand<'tcx>,
        targets: &SwitchTargets,
        target_bb: BasicBlock,
        state: &mut ConditionSet<'a>,
    ) {
        debug_assert_ne!(target_bb, START_BLOCK);
        debug_assert_eq!(self.body.basic_blocks.predecessors()[target_bb].len(), 1);

        let Some(discr) = discr.place() else { return };
        let discr_ty = discr.ty(self.body, self.tcx).ty;
        let Ok(discr_layout) = self.ecx.layout_of(discr_ty) else {
            return;
        };
        let Some(discr) = self.place_value(discr, None) else { return };

        if let Some((value, _)) = targets.iter().find(|&(_, target)| target == target_bb) {
            let Some(value) = ScalarInt::try_from_uint(value, discr_layout.size) else { return };
            debug_assert_eq!(targets.iter().filter(|&(_, target)| target == target_bb).count(), 1);

            // We are inside `target_bb`. Since we have a single predecessor, we know we passed
            // through the `SwitchInt` before arriving here. Therefore, we know that
            // `discr == value`. If one condition can be fulfilled by `discr == value`,
            // that's an opportunity.
            //
            // The TO starts with `target_bb`, which will be added by `find_opportunity`, so we
            // start with an empty bb chain.
            state.register_matches(discr, value, None, &mut self.opportunities);
        } else if let Some((value, _, else_bb)) = targets.as_static_if()
            && target_bb == else_bb
        {
            let Some(value) = ScalarInt::try_from_uint(value, discr_layout.size) else { return };

            // We only know that `discr != value`. That's much weaker information than
            // the equality we had in the previous arm. All we can conclude is that
            // the replacement condition `discr != value` can be threaded, and nothing else.
            for c in state.iter() {
                if c.place == discr && c.value == value && c.polarity == Polarity::Ne {
                    // The TO starts with `target_bb`, which will be added by `find_opportunity`,
                    // so we start with an empty bb chain.
                    self.opportunities.push(c.into_opportunity(None));
                }
            }
        }
    }
}

struct OpportunitySet {
    opportunities: Vec<ThreadingOpportunity>,
    /// For each bb, give the TOs in which it appears. The pair corresponds to the index
    /// in `opportunities` and the index in `ThreadingOpportunity::chain`.
    involving_tos: IndexVec<BasicBlock, Vec<(usize, usize)>>,
    /// Cache the number of predecessors for each block, as we clear the basic block cache..
    predecessors: IndexVec<BasicBlock, usize>,
}

impl OpportunitySet {
    fn new(body: &Body<'_>, opportunities: Vec<ThreadingOpportunity>) -> OpportunitySet {
        let mut involving_tos = IndexVec::from_elem(Vec::new(), &body.basic_blocks);
        for (index, to) in opportunities.iter().enumerate() {
            for (ibb, &bb) in to.chain.iter().enumerate() {
                involving_tos[bb].push((index, ibb));
            }
            involving_tos[to.target].push((index, to.chain.len()));
        }
        let predecessors = predecessor_count(body);
        OpportunitySet { opportunities, involving_tos, predecessors }
    }

    /// Apply the opportunities on the graph.
    fn apply(&mut self, body: &mut Body<'_>) {
        for i in 0..self.opportunities.len() {
            self.apply_once(i, body);
        }
    }

    #[instrument(level = "trace", skip(self, body))]
    fn apply_once(&mut self, index: usize, body: &mut Body<'_>) {
        debug!(?self.predecessors);
        debug!(?self.involving_tos);

        // Check that `predecessors` satisfies its invariant.
        debug_assert_eq!(self.predecessors, predecessor_count(body));

        // Remove the TO from the vector to allow modifying the other ones later.
        let op = &mut self.opportunities[index];
        debug!(?op);
        let op_chain = std::mem::take(&mut op.chain);
        let op_target = op.target;
        debug_assert_eq!(op_chain.len(), op_chain.iter().collect::<FxHashSet<_>>().len());

        let Some((current, chain)) = op_chain.split_first() else { return };
        let basic_blocks = body.basic_blocks.as_mut();

        // Invariant: the control-flow is well-formed at the end of each iteration.
        let mut current = *current;
        for &succ in chain {
            debug!(?current, ?succ);

            // `succ` must be a successor of `current`. If it is not, this means this TO is not
            // satisfiable and a previous TO erased this edge, so we bail out.
            if !basic_blocks[current].terminator().successors().any(|s| s == succ) {
                debug!("impossible");
                return;
            }

            // Fast path: `succ` is only used once, so we can reuse it directly.
            if self.predecessors[succ] == 1 {
                debug!("single");
                current = succ;
                continue;
            }

            let new_succ = basic_blocks.push(basic_blocks[succ].clone());
            debug!(?new_succ);

            // Replace `succ` by `new_succ` where it appears.
            let mut num_edges = 0;
            basic_blocks[current].terminator_mut().successors_mut(|s| {
                if *s == succ {
                    *s = new_succ;
                    num_edges += 1;
                }
            });

            // Update predecessors with the new block.
            let _new_succ = self.predecessors.push(num_edges);
            debug_assert_eq!(new_succ, _new_succ);
            self.predecessors[succ] -= num_edges;
            self.update_predecessor_count(basic_blocks[new_succ].terminator(), Update::Incr);

            // Replace the `current -> succ` edge by `current -> new_succ` in all the following
            // TOs. This is necessary to avoid trying to thread through a non-existing edge. We
            // use `involving_tos` here to avoid traversing the full set of TOs on each iteration.
            let mut new_involved = Vec::new();
            for &(to_index, in_to_index) in &self.involving_tos[current] {
                // That TO has already been applied, do nothing.
                if to_index <= index {
                    continue;
                }

                let other_to = &mut self.opportunities[to_index];
                if other_to.chain.get(in_to_index) != Some(&current) {
                    continue;
                }
                let s = other_to.chain.get_mut(in_to_index + 1).unwrap_or(&mut other_to.target);
                if *s == succ {
                    // `other_to` references the `current -> succ` edge, so replace `succ`.
                    *s = new_succ;
                    new_involved.push((to_index, in_to_index + 1));
                }
            }

            // The TOs that we just updated now reference `new_succ`. Update `involving_tos`
            // in case we need to duplicate an edge starting at `new_succ` later.
            let _new_succ = self.involving_tos.push(new_involved);
            debug_assert_eq!(new_succ, _new_succ);

            current = new_succ;
        }

        let current = &mut basic_blocks[current];
        self.update_predecessor_count(current.terminator(), Update::Decr);
        current.terminator_mut().kind = TerminatorKind::Goto { target: op_target };
        self.predecessors[op_target] += 1;
    }

    fn update_predecessor_count(&mut self, terminator: &Terminator<'_>, incr: Update) {
        match incr {
            Update::Incr => {
                for s in terminator.successors() {
                    self.predecessors[s] += 1;
                }
            }
            Update::Decr => {
                for s in terminator.successors() {
                    self.predecessors[s] -= 1;
                }
            }
        }
    }
}

fn predecessor_count(body: &Body<'_>) -> IndexVec<BasicBlock, usize> {
    let mut predecessors: IndexVec<_, _> =
        body.basic_blocks.predecessors().iter().map(|ps| ps.len()).collect();
    predecessors[START_BLOCK] += 1; // Account for the implicit entry edge.
    predecessors
}

enum Update {
    Incr,
    Decr,
}
