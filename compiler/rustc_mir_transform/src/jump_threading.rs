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
//! We then walk the CFG in post-order transforming the set of conditions.
//! When we find a fulfilling assignment, we record a `ThreadingOpportunity`.
//! All `ThreadingOpportunity`s are applied to the body, by duplicating blocks if required.
//!
//! We try to avoid creating irreducible control-flow by not threading through a loop header.
//!
//! Applying the optimisation can create a lot of new MIR, so we bound the instruction
//! cost by `MAX_COST`.

use std::cell::OnceCell;

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
use rustc_mir_dataflow::value_analysis::{Map, PlaceIndex, TrackElem, ValueIndex};
use rustc_span::DUMMY_SP;
use tracing::{debug, instrument, trace};

use crate::cost_checker::CostChecker;

pub(super) struct JumpThreading;

const MAX_COST: usize = 100;
const MAX_PLACES: usize = 100;

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
            map: Map::new(tcx, body, Some(MAX_PLACES)),
            loop_headers: loop_headers(body),
            entry_states: IndexVec::from_elem(ConditionSet::BOTTOM, &body.basic_blocks),
            opportunities: Vec::new(),
            costs: IndexVec::from_elem(OnceCell::new(), &body.basic_blocks),
        };

        for (bb, bbdata) in traversal::postorder(body) {
            if bbdata.is_cleanup {
                continue;
            }

            let mut state = finder.populate_from_outgoing_edges(bb);
            trace!("output_states[{bb:?}] = {state:?}");

            finder.process_terminator(bb, &mut state);
            trace!("pre_terminator_states[{bb:?}] = {state:?}");

            for stmt in bbdata.statements.iter().rev() {
                if state.is_empty() {
                    break;
                }

                finder.process_statement(bb, stmt, &mut state);

                // When a statement mutates a place, assignments to that place that happen
                // above the mutation cannot fulfill a condition.
                //   _1 = 5 // Whatever happens here, it won't change the result of a `SwitchInt`.
                //   _1 = 6
                if let Some((lhs, tail)) = finder.mutated_statement(stmt) {
                    finder.flood_state(lhs, tail, &mut state);
                }
            }

            trace!("entry_states[{bb:?}] = {state:?}");
            finder.entry_states[bb] = state;
        }

        let opportunities = finder.opportunities;
        debug!(?opportunities);
        if opportunities.is_empty() {
            return;
        }

        // Verify that we do not thread through a loop header.
        for to in opportunities.iter() {
            assert!(to.chain.iter().skip(1).all(|&block| !finder.loop_headers.contains(block)));
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
    loop_headers: DenseBitSet<BasicBlock>,
    /// This stores the state of each visited block on entry,
    /// and the current state of the block being visited.
    // Invariant: for each `bb`, each condition in `entry_states[bb]` has a `chain` that
    // starts with `bb`.
    entry_states: IndexVec<BasicBlock, ConditionSet<'a>>,
    /// We use an arena to avoid cloning the slices when cloning `state`.
    arena: &'a DroplessArena,
    opportunities: Vec<ThreadingOpportunity>,
    /// Pre-computed cost of duplicating each block.
    costs: IndexVec<BasicBlock, OnceCell<usize>>,
}

/// Singly-linked list to represent chains of blocks. This is cheap to copy, and is converted to
/// plain vecs when creating TOs.
#[derive(Copy, Clone)]
struct BBChain<'a> {
    head: BasicBlock,
    tail: Option<&'a BBChain<'a>>,
}

impl std::fmt::Debug for BBChain<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<'a> BBChain<'a> {
    fn single(head: BasicBlock) -> BBChain<'a> {
        BBChain { head, tail: None }
    }

    fn tail_head(self) -> Option<BasicBlock> {
        let tail = self.tail?;
        Some(tail.head)
    }

    fn cons(arena: &'a DroplessArena, head: BasicBlock, tail: BBChain<'a>) -> BBChain<'a> {
        BBChain { head, tail: Some(arena.alloc(tail)) }
    }

    fn to_vec(self) -> Vec<BasicBlock> {
        Vec::from_iter(self.iter())
    }

    fn iter(&self) -> impl Iterator<Item = BasicBlock> {
        return BBChainIter(Some(self));

        struct BBChainIter<'a, 'h>(Option<&'h BBChain<'a>>);

        impl<'a> Iterator for BBChainIter<'a, '_> {
            type Item = BasicBlock;

            fn next(&mut self) -> Option<BasicBlock> {
                let BBChain { head, tail } = self.0?;
                self.0 = *tail;
                Some(*head)
            }
        }
    }
}

/// Represent the following statement. If we can prove that the current local is equal/not-equal
/// to `value`, jump to `target`.
#[derive(Copy, Clone, Debug)]
struct Condition<'a> {
    place: ValueIndex,
    value: ScalarInt,
    polarity: Polarity,
    /// Chain of basic-blocks to traverse from condition fulfilment to target.
    chain: BBChain<'a>,
    target: BasicBlock,
    /// Cumulated cost of duplicating this chain.
    cost: usize,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum Polarity {
    Ne,
    Eq,
}

impl<'a> Condition<'a> {
    fn matches(&self, place: ValueIndex, value: ScalarInt) -> bool {
        self.place == place && (self.value == value) == (self.polarity == Polarity::Eq)
    }

    fn into_opportunity(self) -> ThreadingOpportunity {
        trace!(?self, "registering");
        ThreadingOpportunity { chain: self.chain.to_vec(), target: self.target }
    }
}

#[derive(Clone, Debug)]
struct ConditionSet<'a>(Vec<Condition<'a>>);

impl<'a> HasBottom for ConditionSet<'a> {
    const BOTTOM: Self = ConditionSet(Vec::new());

    fn is_bottom(&self) -> bool {
        self.0.is_empty()
    }
}

impl<'a> ConditionSet<'a> {
    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    fn iter(&self) -> impl Iterator<Item = Condition<'a>> {
        self.0.iter().copied()
    }

    fn iter_matches(
        &self,
        place: ValueIndex,
        value: ScalarInt,
    ) -> impl Iterator<Item = Condition<'a>> {
        self.iter().filter(move |c| c.matches(place, value))
    }

    fn register_matches(
        &self,
        place: ValueIndex,
        value: ScalarInt,
        opportunities: &mut Vec<ThreadingOpportunity>,
    ) {
        self.iter_matches(place, value).for_each(|cond| opportunities.push(cond.into_opportunity()))
    }

    fn retain(&mut self, f: impl Fn(Condition<'a>) -> bool) {
        self.0.retain(|&c| f(c))
    }

    fn retain_mut(&mut self, f: impl Fn(Condition<'a>) -> Option<Condition<'a>>) {
        self.0.retain_mut(|c| {
            if let Some(n) = f(*c) {
                *c = n;
                true
            } else {
                false
            }
        })
    }

    fn for_each_mut(&mut self, f: impl Fn(&mut Condition<'a>)) {
        self.0.iter_mut().for_each(f)
    }
}

impl<'a, 'tcx> TOFinder<'a, 'tcx> {
    /// Construct the condition set for `bb` from the terminator, without executing its effect.
    #[instrument(level = "trace", skip(self))]
    fn populate_from_outgoing_edges(&mut self, bb: BasicBlock) -> ConditionSet<'a> {
        let bbdata = &self.body[bb];

        // This should be the first time we populate `entry_states[bb]`.
        debug_assert!(self.entry_states[bb].is_empty());

        let state_len =
            bbdata.terminator().successors().map(|succ| self.entry_states[succ].0.len()).sum();
        let mut state = Vec::with_capacity(state_len);
        for succ in bbdata.terminator().successors() {
            // Do not thread through loop headers.
            if self.loop_headers.contains(succ) {
                continue;
            }
            state.extend_from_slice(&self.entry_states[succ].0);
        }
        if state.is_empty() {
            return ConditionSet::BOTTOM;
        }
        // Prepend current block to propagated state.
        state.retain_mut(|cond| {
            let head = cond.chain.head;
            cond.chain = BBChain::cons(self.arena, bb, cond.chain);
            // Remove conditions for which the duplication cost is too high.
            // This is required to keep the size of the `ConditionSet` tractable.
            let cost = cond.cost + self.cost(head);
            cond.cost = cost;
            cost <= MAX_COST
        });
        ConditionSet(state)
    }

    fn cost(&self, bb: BasicBlock) -> usize {
        *self.costs[bb].get_or_init(|| {
            let bbdata = &self.body[bb];
            let mut cost = CostChecker::new(self.tcx, self.typing_env, None, self.body);
            cost.visit_basic_block_data(bb, bbdata);
            cost.cost()
        })
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
        state.retain(|c| !places_to_exclude.contains(&c.place));
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
            StatementKind::Assign(box (place, _))
            | StatementKind::Deinit(box place) => Some((place, None)),
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

    #[instrument(level = "trace", skip(self, state))]
    fn process_immediate(
        &mut self,
        bb: BasicBlock,
        lhs: PlaceIndex,
        rhs: ImmTy<'tcx>,
        state: &mut ConditionSet<'a>,
    ) {
        if let Some(lhs) = self.map.value(lhs)
            && let Immediate::Scalar(Scalar::Int(int)) = *rhs
        {
            state.register_matches(lhs, int, &mut self.opportunities)
        }
    }

    /// If we expect `lhs ?= A`, we have an opportunity if we assume `constant == A`.
    #[instrument(level = "trace", skip(self, state))]
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
                    state.register_matches(place, int, &mut self.opportunities)
                }
            },
        );
    }

    #[instrument(level = "trace", skip(self, state))]
    fn process_copy(&mut self, lhs: PlaceIndex, rhs: PlaceIndex, state: &mut ConditionSet<'a>) {
        let mut renames = FxHashMap::default();
        self.map.for_each_value_pair(rhs, lhs, &mut |rhs, lhs| {
            renames.insert(lhs, rhs);
        });
        state.for_each_mut(|c| {
            if let Some(rhs) = renames.get(&c.place) {
                c.place = *rhs
            }
        });
    }

    #[instrument(level = "trace", skip(self, state))]
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
                let Some(rhs) = self.map.find(rhs.as_ref()) else { return };
                self.process_copy(lhs, rhs, state)
            }
        }
    }

    #[instrument(level = "trace", skip(self, state))]
    fn process_assign(
        &mut self,
        bb: BasicBlock,
        lhs_place: &Place<'tcx>,
        rvalue: &Rvalue<'tcx>,
        state: &mut ConditionSet<'a>,
    ) {
        let Some(lhs) = self.map.find(lhs_place.as_ref()) else { return };
        match rvalue {
            Rvalue::Use(operand) => self.process_operand(bb, lhs, operand, state),
            // Transfer the conditions on the copy rhs.
            Rvalue::CopyForDeref(rhs) => {
                let Some(rhs) = self.map.find(rhs.as_ref()) else { return };
                self.process_copy(lhs, rhs, state)
            }
            Rvalue::Discriminant(rhs) => {
                let Some(rhs) = self.map.find_discr(rhs.as_ref()) else { return };
                self.process_copy(lhs, rhs, state)
            }
            // If we expect `lhs ?= A`, we have an opportunity if we assume `constant == A`.
            Rvalue::Aggregate(box kind, operands) => {
                let agg_ty = lhs_place.ty(self.body, self.tcx).ty;
                let lhs = match kind {
                    // Do not support unions.
                    AggregateKind::Adt(.., Some(_)) => return,
                    AggregateKind::Adt(_, variant_index, ..) if agg_ty.is_enum() => {
                        if let Some(discr_target) = self.map.apply(lhs, TrackElem::Discriminant)
                            && let Some(discr_value) = self
                                .ecx
                                .discriminant_for_variant(agg_ty, *variant_index)
                                .discard_err()
                        {
                            self.process_immediate(bb, discr_target, discr_value, state);
                        }
                        if let Some(idx) = self.map.apply(lhs, TrackElem::Variant(*variant_index)) {
                            idx
                        } else {
                            return;
                        }
                    }
                    _ => lhs,
                };
                for (field_index, operand) in operands.iter_enumerated() {
                    if let Some(field) = self.map.apply(lhs, TrackElem::Field(field_index)) {
                        self.process_operand(bb, field, operand, state);
                    }
                }
            }
            // Transfer the conditions on the copy rhs, after inverting the value of the condition.
            Rvalue::UnaryOp(UnOp::Not, Operand::Move(operand) | Operand::Copy(operand)) => {
                let layout = self.ecx.layout_of(operand.ty(self.body, self.tcx).ty).unwrap();
                let Some(lhs) = self.map.value(lhs) else { return };
                let Some(operand) = self.map.find_value(operand.as_ref()) else { return };
                state.retain_mut(|mut c| {
                    if c.place == lhs {
                        let value = self
                            .ecx
                            .unary_op(UnOp::Not, &ImmTy::from_scalar_int(c.value, layout))
                            .discard_err()?
                            .to_scalar_int()
                            .discard_err()?;
                        c.place = operand;
                        c.value = value;
                    }
                    Some(c)
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
                let Some(lhs) = self.map.value(lhs) else { return };
                let Some(operand) = self.map.find_value(operand.as_ref()) else { return };
                let Some(value) = value.const_.try_eval_scalar_int(self.tcx, self.typing_env)
                else {
                    return;
                };
                state.for_each_mut(|c| {
                    if c.place == lhs {
                        let polarity =
                            if c.matches(lhs, equals) { Polarity::Eq } else { Polarity::Ne };
                        c.place = operand;
                        c.value = value;
                        c.polarity = polarity;
                    }
                });
            }

            _ => {}
        }
    }

    #[instrument(level = "trace", skip(self, state))]
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
                let Some(discr_target) = self.map.find_discr(place.as_ref()) else { return };
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
                let Some(place) = self.map.find_value(place.as_ref()) else { return };
                state.register_matches(place, ScalarInt::TRUE, &mut self.opportunities);
            }
            StatementKind::Assign(box (lhs_place, rhs)) => {
                self.process_assign(bb, lhs_place, rhs, state)
            }
            _ => {}
        }
    }

    /// Execute the terminator for block `bb` into state `entry_states[bb]`.
    #[instrument(level = "trace", skip(self, state))]
    fn process_terminator(&mut self, bb: BasicBlock, state: &mut ConditionSet<'a>) {
        let term = self.body.basic_blocks[bb].terminator();
        let place_to_flood = match term.kind {
            // Disallowed during optimizations.
            TerminatorKind::FalseEdge { .. }
            | TerminatorKind::FalseUnwind { .. }
            | TerminatorKind::Yield { .. } => bug!("{term:?} invalid"),
            // Cannot reason about inline asm.
            TerminatorKind::InlineAsm { .. } => {
                *state = ConditionSet::BOTTOM;
                return;
            }
            // `SwitchInt` is handled specially.
            TerminatorKind::SwitchInt { ref discr, ref targets } => {
                return self.process_switch_int(bb, discr, targets, state);
            }
            // These do not modify memory.
            TerminatorKind::UnwindResume
            | TerminatorKind::UnwindTerminate(_)
            | TerminatorKind::Return
            | TerminatorKind::Unreachable
            | TerminatorKind::CoroutineDrop
            // Assertions can be no-op at codegen time, so treat them as such.
            | TerminatorKind::Assert { .. }
            | TerminatorKind::Goto { .. } => None,
            // Flood the overwritten place, and progress through.
            TerminatorKind::Drop { place: destination, .. }
            | TerminatorKind::Call { destination, .. } => Some(destination),
            TerminatorKind::TailCall { .. } => Some(RETURN_PLACE.into()),
        };

        // This terminator modifies `place_to_flood`, cleanup the associated conditions.
        if let Some(place_to_flood) = place_to_flood {
            self.flood_state(place_to_flood, None, state);
        }
    }

    #[instrument(level = "trace", skip(self))]
    fn process_switch_int(
        &mut self,
        bb: BasicBlock,
        discr: &Operand<'tcx>,
        targets: &SwitchTargets,
        state: &mut ConditionSet<'a>,
    ) {
        let Some(discr) = discr.place() else { return };
        let Some(discr_idx) = self.map.find_value(discr.as_ref()) else { return };

        let discr_ty = discr.ty(self.body, self.tcx).ty;
        let Ok(discr_layout) = self.ecx.layout_of(discr_ty) else { return };

        // Attempt to fulfill a condition using an outgoing branch's condition.
        // Only support the case where there are no duplicated outgoing edges.
        if targets.is_distinct() {
            for c in state.iter() {
                if c.place != discr_idx {
                    continue;
                }

                // Invariant from `populate_from_outgoing_edges`.
                debug_assert_eq!(c.chain.head, bb);
                let Some(target_bb) = c.chain.tail_head() else { continue };

                let may_thread = if let Some((branch, _)) =
                    targets.iter().find(|&(_, bb)| bb == target_bb)
                    && let Some(branch) = ScalarInt::try_from_uint(branch, discr_layout.size)
                {
                    // The switch contains a branch `bb -> target_bb` if `discr == branch`.
                    c.matches(discr_idx, branch)
                } else if target_bb == targets.otherwise()
                    && let Ok(value) = c.value.try_to_bits(discr_layout.size)
                {
                    // We only know that `discr` is different from all the constants in the switch.
                    // That's much weaker information than the equality we had in the previous arm.
                    // All we can conclude is that the replacement condition `discr != value` can
                    // be threaded, and nothing else.
                    c.polarity == Polarity::Ne && targets.all_values().contains(&value.into())
                } else {
                    false
                };
                if may_thread {
                    self.opportunities.push(c.into_opportunity());
                }
            }
        }

        // Introduce additional conditions of the form `discr ?= value` for each value in targets.
        let chain = BBChain::single(bb);
        let mk_condition = |value, polarity, target| Condition {
            place: discr_idx,
            value,
            polarity,
            chain,
            target,
            cost: 0,
        };
        if let Some((value, then_, else_)) = targets.as_static_if() {
            // We have an `if`, generate both `discr == value` and `discr != value`.
            let Some(value) = ScalarInt::try_from_uint(value, discr_layout.size) else { return };
            state.0.extend_from_slice(&[
                mk_condition(value, Polarity::Eq, then_),
                mk_condition(value, Polarity::Ne, else_),
            ]);
        } else {
            // We have a general switch and we cannot express `discr != value0 && discr != value1`,
            // so we only generate equality predicates.
            state.0.extend(targets.iter().filter_map(|(value, target)| {
                let value = ScalarInt::try_from_uint(value, discr_layout.size)?;
                Some(mk_condition(value, Polarity::Eq, target))
            }))
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

/// Compute the set of loop headers in the given body. We define a loop header as a block which has
/// at least a predecessor which it dominates. This definition is only correct for reducible CFGs.
/// But if the CFG is already irreducible, there is no point in trying much harder.
/// is already irreducible.
#[instrument(level = "trace", skip(body), ret)]
fn loop_headers(body: &Body<'_>) -> DenseBitSet<BasicBlock> {
    let mut loop_headers = DenseBitSet::new_empty(body.basic_blocks.len());
    let dominators = body.basic_blocks.dominators();
    // Only visit reachable blocks.
    for (bb, bbdata) in traversal::postorder(body) {
        for succ in bbdata.terminator().successors() {
            if dominators.dominates(succ, bb) {
                loop_headers.insert(succ);
            }
        }
    }
    loop_headers
}
