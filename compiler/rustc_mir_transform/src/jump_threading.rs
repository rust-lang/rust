//! A jump threading optimization.
//!
//! This optimization seeks to replace join-then-switch control flow patterns by straight jumps
//!    X = 0                                      X = 0
//! ------------\      /--------              ------------
//!    X = 1     X----X SwitchInt(X)     =>       X = 1
//! ------------/      \--------              ------------
//!
//!
//! This implementation is heavily inspired by the work outlined in [libfirm].
//!
//! The general algorithm proceeds in two phases: (1) walk the CFG backwards to construct a
//! graph of threading conditions, and (2) propagate fulfilled conditions forward by duplicating
//! blocks.
//!
//! # 1. Condition graph construction
//!
//! In this file, we denote as `place ?= value` the existence of a replacement condition
//! on `place` with given `value`, irrespective of the polarity and target of that
//! replacement condition.
//!
//! Inside a block, we associate with each condition `c` a set of targets:
//! - `Goto(target)` if fulfilling `c` changes the terminator into a `Goto { target }`;
//! - `Chain(target, c2)` if fulfilling `c` means that `c2` is fulfilled inside `target`.
//!
//! Before walking a block `bb`, we construct the exit set of condition from its successors.
//! For each condition `c` in a successor `s`, we record that fulfilling `c` in `bb` will fulfill
//! `c` in `s`, as a `Chain(s, c)` condition.
//!
//! When encountering a `switchInt(place) -> [value: bb...]` terminator, we also record a
//! `place == value` condition for each `value`, and associate a `Goto(target)` condition.
//!
//! Then, we walk the statements backwards, transforming the set of conditions along the way,
//! resulting in a set of conditions at the block entry.
//!
//! We try to avoid creating irreducible control-flow by not threading through a loop header.
//!
//! Applying the optimisation can create a lot of new MIR, so we bound the instruction
//! cost by `MAX_COST`.
//!
//! # 2. Block duplication
//!
//! We now have the set of fulfilled conditions inside each block and their targets.
//!
//! For each block `bb` in reverse postorder, we apply in turn the target associated with each
//! fulfilled condition:
//! - for `Goto(target)`, change the terminator of `bb` into a `Goto { target }`;
//! - for `Chain(target, cond)`, duplicate `target` into a new block which fulfills the same
//! conditions and also fulfills `cond`. This is made efficient by maintaining a map of duplicates,
//! `duplicate[(target, cond)]` to avoid cloning blocks multiple times.
//!
//! [libfirm]: <https://pp.ipd.kit.edu/uploads/publikationen/priesner17masterarbeit.pdf>

use itertools::Itertools as _;
use rustc_const_eval::const_eval::DummyMachine;
use rustc_const_eval::interpret::{ImmTy, Immediate, InterpCx, OpTy, Projectable};
use rustc_data_structures::fx::{FxHashMap, FxHashSet, FxIndexSet};
use rustc_index::IndexVec;
use rustc_index::bit_set::{DenseBitSet, GrowableBitSet};
use rustc_middle::bug;
use rustc_middle::mir::interpret::Scalar;
use rustc_middle::mir::visit::Visitor;
use rustc_middle::mir::*;
use rustc_middle::ty::{self, ScalarInt, TyCtxt};
use rustc_mir_dataflow::value_analysis::{
    Map, PlaceCollectionMode, PlaceIndex, TrackElem, ValueIndex,
};
use rustc_span::DUMMY_SP;
use tracing::{debug, instrument, trace};

use crate::cost_checker::CostChecker;

pub(super) struct JumpThreading;

const MAX_COST: u8 = 100;

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
        let mut finder = TOFinder {
            tcx,
            typing_env,
            ecx: InterpCx::new(tcx, DUMMY_SP, typing_env, DummyMachine),
            body,
            map: Map::new(tcx, body, PlaceCollectionMode::OnDemand),
            maybe_loop_headers: loops::maybe_loop_headers(body),
            entry_states: IndexVec::from_elem(ConditionSet::default(), &body.basic_blocks),
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

                finder.process_statement(stmt, &mut state);

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

        let mut entry_states = finder.entry_states;
        simplify_conditions(body, &mut entry_states);
        remove_costly_conditions(tcx, typing_env, body, &mut entry_states);

        if let Some(opportunities) = OpportunitySet::new(body, entry_states) {
            opportunities.apply();
        }
    }

    fn is_required(&self) -> bool {
        false
    }
}

struct TOFinder<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    typing_env: ty::TypingEnv<'tcx>,
    ecx: InterpCx<'tcx, DummyMachine>,
    body: &'a Body<'tcx>,
    map: Map<'tcx>,
    maybe_loop_headers: DenseBitSet<BasicBlock>,
    /// This stores the state of each visited block on entry,
    /// and the current state of the block being visited.
    // Invariant: for each `bb`, each condition in `entry_states[bb]` has a `chain` that
    // starts with `bb`.
    entry_states: IndexVec<BasicBlock, ConditionSet>,
}

rustc_index::newtype_index! {
    #[derive(Ord, PartialOrd)]
    #[debug_format = "_c{}"]
    struct ConditionIndex {}
}

/// Represent the following statement. If we can prove that the current local is equal/not-equal
/// to `value`, jump to `target`.
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
struct Condition {
    place: ValueIndex,
    value: ScalarInt,
    polarity: Polarity,
}

#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
enum Polarity {
    Ne,
    Eq,
}

impl Condition {
    fn matches(&self, place: ValueIndex, value: ScalarInt) -> bool {
        self.place == place && (self.value == value) == (self.polarity == Polarity::Eq)
    }
}

/// Represent the effect of fulfilling a condition.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum EdgeEffect {
    /// If the condition is fulfilled, replace the current block's terminator by a single goto.
    Goto { target: BasicBlock },
    /// If the condition is fulfilled, fulfill the condition `succ_condition` in `succ_block`.
    Chain { succ_block: BasicBlock, succ_condition: ConditionIndex },
}

impl EdgeEffect {
    fn block(self) -> BasicBlock {
        match self {
            EdgeEffect::Goto { target: bb } | EdgeEffect::Chain { succ_block: bb, .. } => bb,
        }
    }

    fn replace_block(&mut self, target: BasicBlock, new_target: BasicBlock) {
        match self {
            EdgeEffect::Goto { target: bb } | EdgeEffect::Chain { succ_block: bb, .. } => {
                if *bb == target {
                    *bb = new_target
                }
            }
        }
    }
}

#[derive(Clone, Debug, Default)]
struct ConditionSet {
    active: Vec<(ConditionIndex, Condition)>,
    fulfilled: Vec<ConditionIndex>,
    targets: IndexVec<ConditionIndex, Vec<EdgeEffect>>,
}

impl ConditionSet {
    fn is_empty(&self) -> bool {
        self.active.is_empty()
    }

    #[tracing::instrument(level = "trace", skip(self))]
    fn push_condition(&mut self, c: Condition, target: BasicBlock) {
        let index = self.targets.push(vec![EdgeEffect::Goto { target }]);
        self.active.push((index, c));
    }

    /// Register fulfilled condition and remove it from the set.
    fn fulfill_if(&mut self, f: impl Fn(Condition, &Vec<EdgeEffect>) -> bool) {
        self.active.retain(|&(index, condition)| {
            let targets = &self.targets[index];
            if f(condition, targets) {
                trace!(?index, ?condition, "fulfill");
                self.fulfilled.push(index);
                false
            } else {
                true
            }
        })
    }

    /// Register fulfilled condition and remove them from the set.
    fn fulfill_matches(&mut self, place: ValueIndex, value: ScalarInt) {
        self.fulfill_if(|c, _| c.matches(place, value))
    }

    fn retain(&mut self, mut f: impl FnMut(Condition) -> bool) {
        self.active.retain(|&(_, c)| f(c))
    }

    fn retain_mut(&mut self, mut f: impl FnMut(Condition) -> Option<Condition>) {
        self.active.retain_mut(|(_, c)| {
            if let Some(new) = f(*c) {
                *c = new;
                true
            } else {
                false
            }
        })
    }

    fn for_each_mut(&mut self, f: impl Fn(&mut Condition)) {
        for (_, c) in &mut self.active {
            f(c)
        }
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

    /// Construct the condition set for `bb` from the terminator, without executing its effect.
    #[instrument(level = "trace", skip(self))]
    fn populate_from_outgoing_edges(&mut self, bb: BasicBlock) -> ConditionSet {
        let bbdata = &self.body[bb];

        // This should be the first time we populate `entry_states[bb]`.
        debug_assert!(self.entry_states[bb].is_empty());

        let state_len =
            bbdata.terminator().successors().map(|succ| self.entry_states[succ].active.len()).sum();
        let mut state = ConditionSet {
            active: Vec::with_capacity(state_len),
            targets: IndexVec::with_capacity(state_len),
            fulfilled: Vec::new(),
        };

        // Use an index-set to deduplicate conditions coming from different successor blocks.
        let mut known_conditions =
            FxIndexSet::with_capacity_and_hasher(state_len, Default::default());
        let mut insert = |condition, succ_block, succ_condition| {
            let (index, new) = known_conditions.insert_full(condition);
            let index = ConditionIndex::from_usize(index);
            if new {
                state.active.push((index, condition));
                let _index = state.targets.push(Vec::new());
                debug_assert_eq!(_index, index);
            }
            let target = EdgeEffect::Chain { succ_block, succ_condition };
            debug_assert!(
                !state.targets[index].contains(&target),
                "duplicate targets for index={index:?} as {target:?} targets={:#?}",
                &state.targets[index],
            );
            state.targets[index].push(target);
        };

        // A given block may have several times the same successor.
        let mut seen = FxHashSet::default();
        for succ in bbdata.terminator().successors() {
            if !seen.insert(succ) {
                continue;
            }

            // Do not thread through loop headers.
            if self.maybe_loop_headers.contains(succ) {
                continue;
            }

            for &(succ_index, cond) in self.entry_states[succ].active.iter() {
                insert(cond, succ, succ_index);
            }
        }

        let num_conditions = known_conditions.len();
        debug_assert_eq!(num_conditions, state.active.len());
        debug_assert_eq!(num_conditions, state.targets.len());
        state.fulfilled.reserve(num_conditions);

        state
    }

    /// Remove all conditions in the state that alias given place.
    fn flood_state(
        &self,
        place: Place<'tcx>,
        extra_elem: Option<TrackElem>,
        state: &mut ConditionSet,
    ) {
        if state.is_empty() {
            return;
        }
        let mut places_to_exclude = FxHashSet::default();
        self.map.for_each_aliasing_place(place.as_ref(), extra_elem, &mut |vi| {
            places_to_exclude.insert(vi);
        });
        trace!(?places_to_exclude, "flood_state");
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

    #[instrument(level = "trace", skip(self, state))]
    fn process_immediate(&mut self, lhs: PlaceIndex, rhs: ImmTy<'tcx>, state: &mut ConditionSet) {
        if let Some(lhs) = self.value(lhs)
            && let Immediate::Scalar(Scalar::Int(int)) = *rhs
        {
            state.fulfill_matches(lhs, int)
        }
    }

    /// If we expect `lhs ?= A`, we have an opportunity if we assume `constant == A`.
    #[instrument(level = "trace", skip(self, state))]
    fn process_constant(
        &mut self,
        lhs: PlaceIndex,
        constant: OpTy<'tcx>,
        state: &mut ConditionSet,
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
                    state.fulfill_matches(place, int)
                }
            },
        );
    }

    #[instrument(level = "trace", skip(self, state))]
    fn process_copy(&mut self, lhs: PlaceIndex, rhs: PlaceIndex, state: &mut ConditionSet) {
        let mut renames = FxHashMap::default();
        self.map.register_copy_tree(
            lhs, // tree to copy
            rhs, // tree to build
            &mut |lhs, rhs| {
                renames.insert(lhs, rhs);
            },
        );
        state.for_each_mut(|c| {
            if let Some(rhs) = renames.get(&c.place) {
                c.place = *rhs
            }
        });
    }

    #[instrument(level = "trace", skip(self, state))]
    fn process_operand(&mut self, lhs: PlaceIndex, rhs: &Operand<'tcx>, state: &mut ConditionSet) {
        match rhs {
            // If we expect `lhs ?= A`, we have an opportunity if we assume `constant == A`.
            Operand::Constant(constant) => {
                let Some(constant) =
                    self.ecx.eval_mir_constant(&constant.const_, constant.span, None).discard_err()
                else {
                    return;
                };
                self.process_constant(lhs, constant, state);
            }
            // Transfer the conditions on the copied rhs.
            Operand::Move(rhs) | Operand::Copy(rhs) => {
                let Some(rhs) = self.place(*rhs, None) else { return };
                self.process_copy(lhs, rhs, state)
            }
            Operand::RuntimeChecks(_) => {}
        }
    }

    #[instrument(level = "trace", skip(self, state))]
    fn process_assign(
        &mut self,
        lhs_place: &Place<'tcx>,
        rvalue: &Rvalue<'tcx>,
        state: &mut ConditionSet,
    ) {
        let Some(lhs) = self.place(*lhs_place, None) else { return };
        match rvalue {
            Rvalue::Use(operand) => self.process_operand(lhs, operand, state),
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
                            self.process_immediate(discr_target, discr_value, state);
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
                    self.process_operand(field, operand, state);
                }
            }
            // Transfer the conditions on the copy rhs, after inverting the value of the condition.
            Rvalue::UnaryOp(UnOp::Not, Operand::Move(operand) | Operand::Copy(operand)) => {
                let layout = self.ecx.layout_of(operand.ty(self.body, self.tcx).ty).unwrap();
                let Some(lhs) = self.value(lhs) else { return };
                let Some(operand) = self.place_value(*operand, None) else { return };
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
                let Some(lhs) = self.value(lhs) else { return };
                let Some(operand) = self.place_value(*operand, None) else { return };
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
    fn process_statement(&mut self, stmt: &Statement<'tcx>, state: &mut ConditionSet) {
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
                self.process_immediate(discr_target, discr, state)
            }
            // If we expect `lhs ?= true`, we have an opportunity if we assume `lhs == true`.
            StatementKind::Intrinsic(box NonDivergingIntrinsic::Assume(
                Operand::Copy(place) | Operand::Move(place),
            )) => {
                let Some(place) = self.place_value(*place, None) else { return };
                state.fulfill_matches(place, ScalarInt::TRUE);
            }
            StatementKind::Assign(box (lhs_place, rhs)) => {
                self.process_assign(lhs_place, rhs, state)
            }
            _ => {}
        }
    }

    /// Execute the terminator for block `bb` into state `entry_states[bb]`.
    #[instrument(level = "trace", skip(self, state))]
    fn process_terminator(&mut self, bb: BasicBlock, state: &mut ConditionSet) {
        let term = self.body.basic_blocks[bb].terminator();
        let place_to_flood = match term.kind {
            // Disallowed during optimizations.
            TerminatorKind::FalseEdge { .. }
            | TerminatorKind::FalseUnwind { .. }
            | TerminatorKind::Yield { .. } => bug!("{term:?} invalid"),
            // Cannot reason about inline asm.
            TerminatorKind::InlineAsm { .. } => {
                state.active.clear();
                return;
            }
            // `SwitchInt` is handled specially.
            TerminatorKind::SwitchInt { ref discr, ref targets } => {
                return self.process_switch_int(discr, targets, state);
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
        discr: &Operand<'tcx>,
        targets: &SwitchTargets,
        state: &mut ConditionSet,
    ) {
        let Some(discr) = discr.place() else { return };
        let Some(discr_idx) = self.place_value(discr, None) else { return };

        let discr_ty = discr.ty(self.body, self.tcx).ty;
        let Ok(discr_layout) = self.ecx.layout_of(discr_ty) else { return };

        // Attempt to fulfill a condition using an outgoing branch's condition.
        // Only support the case where there are no duplicated outgoing edges.
        if targets.is_distinct() {
            for &(index, c) in state.active.iter() {
                if c.place != discr_idx {
                    continue;
                }

                // Set of blocks `t` such that the edge `bb -> t` fulfills `c`.
                let mut edges_fulfilling_condition = FxHashSet::default();

                // On edge `bb -> tgt`, we know that `discr_idx == branch`.
                for (branch, tgt) in targets.iter() {
                    if let Some(branch) = ScalarInt::try_from_uint(branch, discr_layout.size)
                        && c.matches(discr_idx, branch)
                    {
                        edges_fulfilling_condition.insert(tgt);
                    }
                }

                // On edge `bb -> otherwise`, we only know that `discr` is different from all the
                // constants in the switch. That's much weaker information than the equality we
                // had in the previous arm. All we can conclude is that the replacement condition
                // `discr != value` can be threaded, and nothing else.
                if c.polarity == Polarity::Ne
                    && let Ok(value) = c.value.try_to_bits(discr_layout.size)
                    && targets.all_values().contains(&value.into())
                {
                    edges_fulfilling_condition.insert(targets.otherwise());
                }

                // Register that jumping to a `t` fulfills condition `c`.
                // This does *not* mean that `c` is fulfilled in this block: inserting `index` in
                // `fulfilled` is wrong if we have targets that jump to other blocks.
                let condition_targets = &state.targets[index];

                let new_edges: Vec<_> = condition_targets
                    .iter()
                    .copied()
                    .filter(|&target| match target {
                        EdgeEffect::Goto { .. } => false,
                        EdgeEffect::Chain { succ_block, .. } => {
                            edges_fulfilling_condition.contains(&succ_block)
                        }
                    })
                    .collect();

                if new_edges.len() == condition_targets.len() {
                    // If `new_edges == condition_targets`, do not bother creating a new
                    // `ConditionIndex`, we can use the existing one.
                    state.fulfilled.push(index);
                } else {
                    // Fulfilling `index` may thread conditions that we do not want,
                    // so create a brand new index to immediately mark fulfilled.
                    let index = state.targets.push(new_edges);
                    state.fulfilled.push(index);
                }
            }
        }

        // Introduce additional conditions of the form `discr ?= value` for each value in targets.
        let mut mk_condition = |value, polarity, target| {
            let c = Condition { place: discr_idx, value, polarity };
            state.push_condition(c, target);
        };
        if let Some((value, then_, else_)) = targets.as_static_if() {
            // We have an `if`, generate both `discr == value` and `discr != value`.
            let Some(value) = ScalarInt::try_from_uint(value, discr_layout.size) else { return };
            mk_condition(value, Polarity::Eq, then_);
            mk_condition(value, Polarity::Ne, else_);
        } else {
            // We have a general switch and we cannot express `discr != value0 && discr != value1`,
            // so we only generate equality predicates.
            for (value, target) in targets.iter() {
                if let Some(value) = ScalarInt::try_from_uint(value, discr_layout.size) {
                    mk_condition(value, Polarity::Eq, target);
                }
            }
        }
    }
}

/// Propagate fulfilled conditions forward in the CFG to reduce the amount of duplication.
#[instrument(level = "debug", skip(body, entry_states))]
fn simplify_conditions(body: &Body<'_>, entry_states: &mut IndexVec<BasicBlock, ConditionSet>) {
    let basic_blocks = &body.basic_blocks;
    let reverse_postorder = basic_blocks.reverse_postorder();

    // Start by computing the number of *incoming edges* for each block.
    // We do not use the cached `basic_blocks.predecessors` as we only want reachable predecessors.
    let mut predecessors = IndexVec::from_elem(0, &entry_states);
    predecessors[START_BLOCK] = 1; // Account for the implicit entry edge.
    for &bb in reverse_postorder {
        let term = basic_blocks[bb].terminator();
        for s in term.successors() {
            predecessors[s] += 1;
        }
    }

    // Compute the number of edges into each block that carry each condition.
    let mut fulfill_in_pred_count = IndexVec::from_fn_n(
        |bb: BasicBlock| IndexVec::from_elem_n(0, entry_states[bb].targets.len()),
        entry_states.len(),
    );

    // By traversing in RPO, we increase the likelihood to visit predecessors before successors.
    for &bb in reverse_postorder {
        let preds = predecessors[bb];
        trace!(?bb, ?preds);

        // We have removed all the input edges towards this block. Just skip visiting it.
        if preds == 0 {
            continue;
        }

        let state = &mut entry_states[bb];
        trace!(?state);

        // Conditions that are fulfilled in all the predecessors, are fulfilled in `bb`.
        trace!(fulfilled_count = ?fulfill_in_pred_count[bb]);
        for (condition, &cond_preds) in fulfill_in_pred_count[bb].iter_enumerated() {
            if cond_preds == preds {
                trace!(?condition);
                state.fulfilled.push(condition);
            }
        }

        // We want to count how many times each condition is fulfilled,
        // so ensure we are not counting the same edge twice.
        let mut targets: Vec<_> = state
            .fulfilled
            .iter()
            .flat_map(|&index| state.targets[index].iter().copied())
            .collect();
        targets.sort();
        targets.dedup();
        trace!(?targets);

        // We may modify the set of successors by applying edges, so track them here.
        let mut successors = basic_blocks[bb].terminator().successors().collect::<Vec<_>>();

        targets.reverse();
        while let Some(target) = targets.pop() {
            match target {
                EdgeEffect::Goto { target } => {
                    // We update the count of predecessors. If target or any successor has not been
                    // processed yet, this increases the likelihood we find something relevant.
                    predecessors[target] += 1;
                    for &s in successors.iter() {
                        predecessors[s] -= 1;
                    }
                    // Only process edges that still exist.
                    targets.retain(|t| t.block() == target);
                    successors.clear();
                    successors.push(target);
                }
                EdgeEffect::Chain { succ_block, succ_condition } => {
                    // `predecessors` is the number of incoming *edges* in each block.
                    // Count the number of edges that apply `succ_condition` into `succ_block`.
                    let count = successors.iter().filter(|&&s| s == succ_block).count();
                    fulfill_in_pred_count[succ_block][succ_condition] += count;
                }
            }
        }
    }
}

#[instrument(level = "debug", skip(tcx, typing_env, body, entry_states))]
fn remove_costly_conditions<'tcx>(
    tcx: TyCtxt<'tcx>,
    typing_env: ty::TypingEnv<'tcx>,
    body: &Body<'tcx>,
    entry_states: &mut IndexVec<BasicBlock, ConditionSet>,
) {
    let basic_blocks = &body.basic_blocks;

    let mut costs = IndexVec::from_elem(None, basic_blocks);
    let mut cost = |bb: BasicBlock| -> u8 {
        let c = *costs[bb].get_or_insert_with(|| {
            let bbdata = &basic_blocks[bb];
            let mut cost = CostChecker::new(tcx, typing_env, None, body);
            cost.visit_basic_block_data(bb, bbdata);
            cost.cost().try_into().unwrap_or(MAX_COST)
        });
        trace!("cost[{bb:?}] = {c}");
        c
    };

    // Initialize costs with `MAX_COST`: if we have a cycle, the cyclic `bb` has infinite costs.
    let mut condition_cost = IndexVec::from_fn_n(
        |bb: BasicBlock| IndexVec::from_elem_n(MAX_COST, entry_states[bb].targets.len()),
        entry_states.len(),
    );

    let reverse_postorder = basic_blocks.reverse_postorder();

    for &bb in reverse_postorder.iter().rev() {
        let state = &entry_states[bb];
        trace!(?bb, ?state);

        let mut current_costs = IndexVec::from_elem(0u8, &state.targets);

        for (condition, targets) in state.targets.iter_enumerated() {
            for &target in targets {
                match target {
                    // A `Goto` has cost 0.
                    EdgeEffect::Goto { .. } => {}
                    // Chaining into an already-fulfilled condition is nop.
                    EdgeEffect::Chain { succ_block, succ_condition }
                        if entry_states[succ_block].fulfilled.contains(&succ_condition) => {}
                    // When chaining, use `cost[succ_block][succ_condition] + cost(succ_block)`.
                    EdgeEffect::Chain { succ_block, succ_condition } => {
                        // Cost associated with duplicating `succ_block`.
                        let duplication_cost = cost(succ_block);
                        // Cost associated with the rest of the chain.
                        let target_cost =
                            *condition_cost[succ_block].get(succ_condition).unwrap_or(&MAX_COST);
                        let cost = current_costs[condition]
                            .saturating_add(duplication_cost)
                            .saturating_add(target_cost);
                        trace!(?condition, ?succ_block, ?duplication_cost, ?target_cost);
                        current_costs[condition] = cost;
                    }
                }
            }
        }

        trace!("condition_cost[{bb:?}] = {:?}", current_costs);
        condition_cost[bb] = current_costs;
    }

    trace!(?condition_cost);

    for &bb in reverse_postorder {
        for (index, targets) in entry_states[bb].targets.iter_enumerated_mut() {
            if condition_cost[bb][index] >= MAX_COST {
                trace!(?bb, ?index, ?targets, c = ?condition_cost[bb][index], "remove");
                targets.clear()
            }
        }
    }
}

struct OpportunitySet<'a, 'tcx> {
    basic_blocks: &'a mut IndexVec<BasicBlock, BasicBlockData<'tcx>>,
    entry_states: IndexVec<BasicBlock, ConditionSet>,
    /// Cache duplicated block. When cloning a basic block `bb` to fulfill a condition `c`,
    /// record the target of this `bb with c` edge.
    duplicates: FxHashMap<(BasicBlock, ConditionIndex), BasicBlock>,
}

impl<'a, 'tcx> OpportunitySet<'a, 'tcx> {
    fn new(
        body: &'a mut Body<'tcx>,
        mut entry_states: IndexVec<BasicBlock, ConditionSet>,
    ) -> Option<OpportunitySet<'a, 'tcx>> {
        trace!(def_id = ?body.source.def_id(), "apply");

        if entry_states.iter().all(|state| state.fulfilled.is_empty()) {
            return None;
        }

        // Free some memory, because we will need to clone condition sets.
        for state in entry_states.iter_mut() {
            state.active = Default::default();
        }
        let duplicates = Default::default();
        let basic_blocks = body.basic_blocks.as_mut();
        Some(OpportunitySet { basic_blocks, entry_states, duplicates })
    }

    /// Apply the opportunities on the graph.
    #[instrument(level = "debug", skip(self))]
    fn apply(mut self) {
        let mut worklist = Vec::with_capacity(self.basic_blocks.len());
        worklist.push(START_BLOCK);

        // Use a `GrowableBitSet` and not a `DenseBitSet` as we are adding blocks.
        let mut visited = GrowableBitSet::with_capacity(self.basic_blocks.len());

        while let Some(bb) = worklist.pop() {
            if !visited.insert(bb) {
                continue;
            }

            self.apply_once(bb);

            // `apply_once` may have modified the terminator of `bb`.
            // Only visit actual successors.
            worklist.extend(self.basic_blocks[bb].terminator().successors());
        }
    }

    /// Apply the opportunities on `bb`.
    #[instrument(level = "debug", skip(self))]
    fn apply_once(&mut self, bb: BasicBlock) {
        let state = &mut self.entry_states[bb];
        trace!(?state);

        // We are modifying the `bb` in-place. Once a `EdgeEffect` has been applied,
        // it does not need to be applied again.
        let mut targets: Vec<_> = state
            .fulfilled
            .iter()
            .flat_map(|&index| std::mem::take(&mut state.targets[index]))
            .collect();
        targets.sort();
        targets.dedup();
        trace!(?targets);

        // Use a while-pop to allow modifying `targets` from inside the loop.
        targets.reverse();
        while let Some(target) = targets.pop() {
            debug!(?target);
            trace!(term = ?self.basic_blocks[bb].terminator().kind);

            // By construction, `target.block()` is a successor of `bb`.
            // When applying targets, we may change the set of successors.
            // The match below updates the set of targets for consistency.
            debug_assert!(
                self.basic_blocks[bb].terminator().successors().contains(&target.block()),
                "missing {target:?} in successors for {bb:?}, term={:?}",
                self.basic_blocks[bb].terminator(),
            );

            match target {
                EdgeEffect::Goto { target } => {
                    self.apply_goto(bb, target);

                    // We now have `target` as single successor. Drop all other target blocks.
                    targets.retain(|t| t.block() == target);
                    // Also do this on targets that may be applied by a duplicate of `bb`.
                    for ts in self.entry_states[bb].targets.iter_mut() {
                        ts.retain(|t| t.block() == target);
                    }
                }
                EdgeEffect::Chain { succ_block, succ_condition } => {
                    let new_succ_block = self.apply_chain(bb, succ_block, succ_condition);

                    // We have a new name for `target`, ensure it is correctly applied.
                    if let Some(new_succ_block) = new_succ_block {
                        for t in targets.iter_mut() {
                            t.replace_block(succ_block, new_succ_block)
                        }
                        // Also do this on targets that may be applied by a duplicate of `bb`.
                        for t in
                            self.entry_states[bb].targets.iter_mut().flat_map(|ts| ts.iter_mut())
                        {
                            t.replace_block(succ_block, new_succ_block)
                        }
                    }
                }
            }

            trace!(post_term = ?self.basic_blocks[bb].terminator().kind);
        }
    }

    #[instrument(level = "debug", skip(self))]
    fn apply_goto(&mut self, bb: BasicBlock, target: BasicBlock) {
        self.basic_blocks[bb].terminator_mut().kind = TerminatorKind::Goto { target };
    }

    #[instrument(level = "debug", skip(self), ret)]
    fn apply_chain(
        &mut self,
        bb: BasicBlock,
        target: BasicBlock,
        condition: ConditionIndex,
    ) -> Option<BasicBlock> {
        if self.entry_states[target].fulfilled.contains(&condition) {
            // `target` already fulfills `condition`, so we do not need to thread anything.
            trace!("fulfilled");
            return None;
        }

        // We may be tempted to modify `target` in-place to avoid a clone. This is wrong.
        // We may still have edges from other blocks to `target` that have not been created yet.
        // For instance because we may be threading an edge coming from `bb`,
        // or `target` may be a block duplicate for which we may still create predecessors.

        let new_target = *self.duplicates.entry((target, condition)).or_insert_with(|| {
            // If we already have a duplicate of `target` which fulfills `condition`, reuse it.
            // Otherwise, we clone a new bb to such ends.
            let new_target = self.basic_blocks.push(self.basic_blocks[target].clone());
            trace!(?target, ?new_target, ?condition, "clone");

            // By definition, `new_target` fulfills the same condition as `target`, with
            // `condition` added.
            let mut condition_set = self.entry_states[target].clone();
            condition_set.fulfilled.push(condition);
            let _new_target = self.entry_states.push(condition_set);
            debug_assert_eq!(new_target, _new_target);

            new_target
        });
        trace!(?target, ?new_target, ?condition, "reuse");

        // Replace `target` by `new_target` where it appears.
        // This changes exactly `direct_count` edges.
        self.basic_blocks[bb].terminator_mut().successors_mut(|s| {
            if *s == target {
                *s = new_target;
            }
        });

        Some(new_target)
    }
}
