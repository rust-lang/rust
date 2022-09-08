//! A framework that can express both [gen-kill] and generic dataflow problems.
//!
//! To use this framework, implement either the [`Analysis`] or the
//! [`GenKillAnalysis`] trait. If your transfer function can be expressed with only gen/kill
//! operations, prefer `GenKillAnalysis` since it will run faster while iterating to fixpoint. The
//! `impls` module contains several examples of gen/kill dataflow analyses.
//!
//! Create an `Engine` for your analysis using the `into_engine` method on the `Analysis` trait,
//! then call `iterate_to_fixpoint`. From there, you can use a `ResultsCursor` to inspect the
//! fixpoint solution to your dataflow problem, or implement the `ResultsVisitor` interface and use
//! `visit_results`. The following example uses the `ResultsCursor` approach.
//!
//! ```ignore (cross-crate-imports)
//! use rustc_const_eval::dataflow::Analysis; // Makes `into_engine` available.
//!
//! fn do_my_analysis(tcx: TyCtxt<'tcx>, body: &mir::Body<'tcx>) {
//!     let analysis = MyAnalysis::new()
//!         .into_engine(tcx, body)
//!         .iterate_to_fixpoint()
//!         .into_results_cursor(body);
//!
//!     // Print the dataflow state *after* each statement in the start block.
//!     for (_, statement_index) in body.block_data[START_BLOCK].statements.iter_enumerated() {
//!         cursor.seek_after(Location { block: START_BLOCK, statement_index });
//!         let state = cursor.get();
//!         println!("{:?}", state);
//!     }
//! }
//! ```
//!
//! [gen-kill]: https://en.wikipedia.org/wiki/Data-flow_analysis#Bit_vector_problems

use std::cmp::Ordering;

use rustc_index::bit_set::{BitSet, ChunkedBitSet, HybridBitSet};
use rustc_index::vec::Idx;
use rustc_middle::mir::{self, BasicBlock, Location};
use rustc_middle::ty::TyCtxt;

mod cursor;
mod direction;
mod engine;
pub mod fmt;
pub mod graphviz;
pub mod lattice;
mod visitor;

pub use self::cursor::{ResultsCursor, ResultsRefCursor};
pub use self::direction::{Backward, Direction, Forward};
pub use self::engine::{Engine, Results};
pub use self::lattice::{JoinSemiLattice, MeetSemiLattice};
pub use self::visitor::{visit_results, ResultsVisitable, ResultsVisitor};

/// Analysis domains are all bitsets of various kinds. This trait holds
/// operations needed by all of them.
pub trait BitSetExt<T> {
    fn domain_size(&self) -> usize;
    fn contains(&self, elem: T) -> bool;
    fn union(&mut self, other: &HybridBitSet<T>);
    fn subtract(&mut self, other: &HybridBitSet<T>);
}

impl<T: Idx> BitSetExt<T> for BitSet<T> {
    fn domain_size(&self) -> usize {
        self.domain_size()
    }

    fn contains(&self, elem: T) -> bool {
        self.contains(elem)
    }

    fn union(&mut self, other: &HybridBitSet<T>) {
        self.union(other);
    }

    fn subtract(&mut self, other: &HybridBitSet<T>) {
        self.subtract(other);
    }
}

impl<T: Idx> BitSetExt<T> for ChunkedBitSet<T> {
    fn domain_size(&self) -> usize {
        self.domain_size()
    }

    fn contains(&self, elem: T) -> bool {
        self.contains(elem)
    }

    fn union(&mut self, other: &HybridBitSet<T>) {
        self.union(other);
    }

    fn subtract(&mut self, other: &HybridBitSet<T>) {
        self.subtract(other);
    }
}

/// Defines the domain of a dataflow problem.
///
/// This trait specifies the lattice on which this analysis operates (the domain) as well as its
/// initial value at the entry point of each basic block.
pub trait AnalysisDomain<'tcx> {
    /// The type that holds the dataflow state at any given point in the program.
    type Domain: Clone + JoinSemiLattice;

    /// The direction of this analysis. Either `Forward` or `Backward`.
    type Direction: Direction = Forward;

    /// A descriptive name for this analysis. Used only for debugging.
    ///
    /// This name should be brief and contain no spaces, periods or other characters that are not
    /// suitable as part of a filename.
    const NAME: &'static str;

    /// Returns the initial value of the dataflow state upon entry to each basic block.
    fn bottom_value(&self, body: &mir::Body<'tcx>) -> Self::Domain;

    /// Mutates the initial value of the dataflow state upon entry to the `START_BLOCK`.
    ///
    /// For backward analyses, initial state (besides the bottom value) is not yet supported. Trying
    /// to mutate the initial state will result in a panic.
    //
    // FIXME: For backward dataflow analyses, the initial state should be applied to every basic
    // block where control flow could exit the MIR body (e.g., those terminated with `return` or
    // `resume`). It's not obvious how to handle `yield` points in generators, however.
    fn initialize_start_block(&self, body: &mir::Body<'tcx>, state: &mut Self::Domain);
}

/// A dataflow problem with an arbitrarily complex transfer function.
///
/// # Convergence
///
/// When implementing this trait directly (not via [`GenKillAnalysis`]), it's possible to choose a
/// transfer function such that the analysis does not reach fixpoint. To guarantee convergence,
/// your transfer functions must maintain the following invariant:
///
/// > If the dataflow state **before** some point in the program changes to be greater
/// than the prior state **before** that point, the dataflow state **after** that point must
/// also change to be greater than the prior state **after** that point.
///
/// This invariant guarantees that the dataflow state at a given point in the program increases
/// monotonically until fixpoint is reached. Note that this monotonicity requirement only applies
/// to the same point in the program at different points in time. The dataflow state at a given
/// point in the program may or may not be greater than the state at any preceding point.
pub trait Analysis<'tcx>: AnalysisDomain<'tcx> {
    /// Updates the current dataflow state with the effect of evaluating a statement.
    fn apply_statement_effect(
        &self,
        state: &mut Self::Domain,
        statement: &mir::Statement<'tcx>,
        location: Location,
    );

    /// Updates the current dataflow state with an effect that occurs immediately *before* the
    /// given statement.
    ///
    /// This method is useful if the consumer of the results of this analysis only needs to observe
    /// *part* of the effect of a statement (e.g. for two-phase borrows). As a general rule,
    /// analyses should not implement this without also implementing `apply_statement_effect`.
    fn apply_before_statement_effect(
        &self,
        _state: &mut Self::Domain,
        _statement: &mir::Statement<'tcx>,
        _location: Location,
    ) {
    }

    /// Updates the current dataflow state with the effect of evaluating a terminator.
    ///
    /// The effect of a successful return from a `Call` terminator should **not** be accounted for
    /// in this function. That should go in `apply_call_return_effect`. For example, in the
    /// `InitializedPlaces` analyses, the return place for a function call is not marked as
    /// initialized here.
    fn apply_terminator_effect(
        &self,
        state: &mut Self::Domain,
        terminator: &mir::Terminator<'tcx>,
        location: Location,
    );

    /// Updates the current dataflow state with an effect that occurs immediately *before* the
    /// given terminator.
    ///
    /// This method is useful if the consumer of the results of this analysis needs only to observe
    /// *part* of the effect of a terminator (e.g. for two-phase borrows). As a general rule,
    /// analyses should not implement this without also implementing `apply_terminator_effect`.
    fn apply_before_terminator_effect(
        &self,
        _state: &mut Self::Domain,
        _terminator: &mir::Terminator<'tcx>,
        _location: Location,
    ) {
    }

    /* Edge-specific effects */

    /// Updates the current dataflow state with the effect of a successful return from a `Call`
    /// terminator.
    ///
    /// This is separate from `apply_terminator_effect` to properly track state across unwind
    /// edges.
    fn apply_call_return_effect(
        &self,
        state: &mut Self::Domain,
        block: BasicBlock,
        return_places: CallReturnPlaces<'_, 'tcx>,
    );

    /// Updates the current dataflow state with the effect of resuming from a `Yield` terminator.
    ///
    /// This is similar to `apply_call_return_effect` in that it only takes place after the
    /// generator is resumed, not when it is dropped.
    ///
    /// By default, no effects happen.
    fn apply_yield_resume_effect(
        &self,
        _state: &mut Self::Domain,
        _resume_block: BasicBlock,
        _resume_place: mir::Place<'tcx>,
    ) {
    }

    /// Updates the current dataflow state with the effect of taking a particular branch in a
    /// `SwitchInt` terminator.
    ///
    /// Unlike the other edge-specific effects, which are allowed to mutate `Self::Domain`
    /// directly, overriders of this method must pass a callback to
    /// `SwitchIntEdgeEffects::apply`. The callback will be run once for each outgoing edge and
    /// will have access to the dataflow state that will be propagated along that edge.
    ///
    /// This interface is somewhat more complex than the other visitor-like "effect" methods.
    /// However, it is both more ergonomic—callers don't need to recompute or cache information
    /// about a given `SwitchInt` terminator for each one of its edges—and more efficient—the
    /// engine doesn't need to clone the exit state for a block unless
    /// `SwitchIntEdgeEffects::apply` is actually called.
    fn apply_switch_int_edge_effects(
        &self,
        _block: BasicBlock,
        _discr: &mir::Operand<'tcx>,
        _apply_edge_effects: &mut impl SwitchIntEdgeEffects<Self::Domain>,
    ) {
    }

    /* Extension methods */

    /// Creates an `Engine` to find the fixpoint for this dataflow problem.
    ///
    /// You shouldn't need to override this outside this module, since the combination of the
    /// default impl and the one for all `A: GenKillAnalysis` will do the right thing.
    /// Its purpose is to enable method chaining like so:
    ///
    /// ```ignore (cross-crate-imports)
    /// let results = MyAnalysis::new(tcx, body)
    ///     .into_engine(tcx, body, def_id)
    ///     .iterate_to_fixpoint()
    ///     .into_results_cursor(body);
    /// ```
    #[inline]
    fn into_engine<'mir>(
        self,
        tcx: TyCtxt<'tcx>,
        body: &'mir mir::Body<'tcx>,
    ) -> Engine<'mir, 'tcx, Self>
    where
        Self: Sized,
    {
        Engine::new_generic(tcx, body, self)
    }
}

/// A gen/kill dataflow problem.
///
/// Each method in this trait has a corresponding one in `Analysis`. However, these methods only
/// allow modification of the dataflow state via "gen" and "kill" operations. By defining transfer
/// functions for each statement in this way, the transfer function for an entire basic block can
/// be computed efficiently.
///
/// `Analysis` is automatically implemented for all implementers of `GenKillAnalysis`.
pub trait GenKillAnalysis<'tcx>: Analysis<'tcx> {
    type Idx: Idx;

    /// See `Analysis::apply_statement_effect`.
    fn statement_effect(
        &self,
        trans: &mut impl GenKill<Self::Idx>,
        statement: &mir::Statement<'tcx>,
        location: Location,
    );

    /// See `Analysis::apply_before_statement_effect`.
    fn before_statement_effect(
        &self,
        _trans: &mut impl GenKill<Self::Idx>,
        _statement: &mir::Statement<'tcx>,
        _location: Location,
    ) {
    }

    /// See `Analysis::apply_terminator_effect`.
    fn terminator_effect(
        &self,
        trans: &mut impl GenKill<Self::Idx>,
        terminator: &mir::Terminator<'tcx>,
        location: Location,
    );

    /// See `Analysis::apply_before_terminator_effect`.
    fn before_terminator_effect(
        &self,
        _trans: &mut impl GenKill<Self::Idx>,
        _terminator: &mir::Terminator<'tcx>,
        _location: Location,
    ) {
    }

    /* Edge-specific effects */

    /// See `Analysis::apply_call_return_effect`.
    fn call_return_effect(
        &self,
        trans: &mut impl GenKill<Self::Idx>,
        block: BasicBlock,
        return_places: CallReturnPlaces<'_, 'tcx>,
    );

    /// See `Analysis::apply_yield_resume_effect`.
    fn yield_resume_effect(
        &self,
        _trans: &mut impl GenKill<Self::Idx>,
        _resume_block: BasicBlock,
        _resume_place: mir::Place<'tcx>,
    ) {
    }

    /// See `Analysis::apply_switch_int_edge_effects`.
    fn switch_int_edge_effects<G: GenKill<Self::Idx>>(
        &self,
        _block: BasicBlock,
        _discr: &mir::Operand<'tcx>,
        _edge_effects: &mut impl SwitchIntEdgeEffects<G>,
    ) {
    }
}

impl<'tcx, A> Analysis<'tcx> for A
where
    A: GenKillAnalysis<'tcx>,
    A::Domain: GenKill<A::Idx> + BitSetExt<A::Idx>,
{
    fn apply_statement_effect(
        &self,
        state: &mut A::Domain,
        statement: &mir::Statement<'tcx>,
        location: Location,
    ) {
        self.statement_effect(state, statement, location);
    }

    fn apply_before_statement_effect(
        &self,
        state: &mut A::Domain,
        statement: &mir::Statement<'tcx>,
        location: Location,
    ) {
        self.before_statement_effect(state, statement, location);
    }

    fn apply_terminator_effect(
        &self,
        state: &mut A::Domain,
        terminator: &mir::Terminator<'tcx>,
        location: Location,
    ) {
        self.terminator_effect(state, terminator, location);
    }

    fn apply_before_terminator_effect(
        &self,
        state: &mut A::Domain,
        terminator: &mir::Terminator<'tcx>,
        location: Location,
    ) {
        self.before_terminator_effect(state, terminator, location);
    }

    /* Edge-specific effects */

    fn apply_call_return_effect(
        &self,
        state: &mut A::Domain,
        block: BasicBlock,
        return_places: CallReturnPlaces<'_, 'tcx>,
    ) {
        self.call_return_effect(state, block, return_places);
    }

    fn apply_yield_resume_effect(
        &self,
        state: &mut A::Domain,
        resume_block: BasicBlock,
        resume_place: mir::Place<'tcx>,
    ) {
        self.yield_resume_effect(state, resume_block, resume_place);
    }

    fn apply_switch_int_edge_effects(
        &self,
        block: BasicBlock,
        discr: &mir::Operand<'tcx>,
        edge_effects: &mut impl SwitchIntEdgeEffects<A::Domain>,
    ) {
        self.switch_int_edge_effects(block, discr, edge_effects);
    }

    /* Extension methods */
    #[inline]
    fn into_engine<'mir>(
        self,
        tcx: TyCtxt<'tcx>,
        body: &'mir mir::Body<'tcx>,
    ) -> Engine<'mir, 'tcx, Self>
    where
        Self: Sized,
    {
        Engine::new_gen_kill(tcx, body, self)
    }
}

/// The legal operations for a transfer function in a gen/kill problem.
///
/// This abstraction exists because there are two different contexts in which we call the methods in
/// `GenKillAnalysis`. Sometimes we need to store a single transfer function that can be efficiently
/// applied multiple times, such as when computing the cumulative transfer function for each block.
/// These cases require a `GenKillSet`, which in turn requires two `BitSet`s of storage. Oftentimes,
/// however, we only need to apply an effect once. In *these* cases, it is more efficient to pass the
/// `BitSet` representing the state vector directly into the `*_effect` methods as opposed to
/// building up a `GenKillSet` and then throwing it away.
pub trait GenKill<T> {
    /// Inserts `elem` into the state vector.
    fn gen(&mut self, elem: T);

    /// Removes `elem` from the state vector.
    fn kill(&mut self, elem: T);

    /// Calls `gen` for each element in `elems`.
    fn gen_all(&mut self, elems: impl IntoIterator<Item = T>) {
        for elem in elems {
            self.gen(elem);
        }
    }

    /// Calls `kill` for each element in `elems`.
    fn kill_all(&mut self, elems: impl IntoIterator<Item = T>) {
        for elem in elems {
            self.kill(elem);
        }
    }
}

/// Stores a transfer function for a gen/kill problem.
///
/// Calling `gen`/`kill` on a `GenKillSet` will "build up" a transfer function so that it can be
/// applied multiple times efficiently. When there are multiple calls to `gen` and/or `kill` for
/// the same element, the most recent one takes precedence.
#[derive(Clone)]
pub struct GenKillSet<T> {
    gen: HybridBitSet<T>,
    kill: HybridBitSet<T>,
}

impl<T: Idx> GenKillSet<T> {
    /// Creates a new transfer function that will leave the dataflow state unchanged.
    pub fn identity(universe: usize) -> Self {
        GenKillSet {
            gen: HybridBitSet::new_empty(universe),
            kill: HybridBitSet::new_empty(universe),
        }
    }

    pub fn apply(&self, state: &mut impl BitSetExt<T>) {
        state.union(&self.gen);
        state.subtract(&self.kill);
    }
}

impl<T: Idx> GenKill<T> for GenKillSet<T> {
    fn gen(&mut self, elem: T) {
        self.gen.insert(elem);
        self.kill.remove(elem);
    }

    fn kill(&mut self, elem: T) {
        self.kill.insert(elem);
        self.gen.remove(elem);
    }
}

impl<T: Idx> GenKill<T> for BitSet<T> {
    fn gen(&mut self, elem: T) {
        self.insert(elem);
    }

    fn kill(&mut self, elem: T) {
        self.remove(elem);
    }
}

impl<T: Idx> GenKill<T> for ChunkedBitSet<T> {
    fn gen(&mut self, elem: T) {
        self.insert(elem);
    }

    fn kill(&mut self, elem: T) {
        self.remove(elem);
    }
}

impl<T: Idx> GenKill<T> for lattice::Dual<BitSet<T>> {
    fn gen(&mut self, elem: T) {
        self.0.insert(elem);
    }

    fn kill(&mut self, elem: T) {
        self.0.remove(elem);
    }
}

// NOTE: DO NOT CHANGE VARIANT ORDER. The derived `Ord` impls rely on the current order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Effect {
    /// The "before" effect (e.g., `apply_before_statement_effect`) for a statement (or
    /// terminator).
    Before,

    /// The "primary" effect (e.g., `apply_statement_effect`) for a statement (or terminator).
    Primary,
}

impl Effect {
    pub const fn at_index(self, statement_index: usize) -> EffectIndex {
        EffectIndex { effect: self, statement_index }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EffectIndex {
    statement_index: usize,
    effect: Effect,
}

impl EffectIndex {
    fn next_in_forward_order(self) -> Self {
        match self.effect {
            Effect::Before => Effect::Primary.at_index(self.statement_index),
            Effect::Primary => Effect::Before.at_index(self.statement_index + 1),
        }
    }

    fn next_in_backward_order(self) -> Self {
        match self.effect {
            Effect::Before => Effect::Primary.at_index(self.statement_index),
            Effect::Primary => Effect::Before.at_index(self.statement_index - 1),
        }
    }

    /// Returns `true` if the effect at `self` should be applied earlier than the effect at `other`
    /// in forward order.
    fn precedes_in_forward_order(self, other: Self) -> bool {
        let ord = self
            .statement_index
            .cmp(&other.statement_index)
            .then_with(|| self.effect.cmp(&other.effect));
        ord == Ordering::Less
    }

    /// Returns `true` if the effect at `self` should be applied earlier than the effect at `other`
    /// in backward order.
    fn precedes_in_backward_order(self, other: Self) -> bool {
        let ord = other
            .statement_index
            .cmp(&self.statement_index)
            .then_with(|| self.effect.cmp(&other.effect));
        ord == Ordering::Less
    }
}

pub struct SwitchIntTarget {
    pub value: Option<u128>,
    pub target: BasicBlock,
}

/// A type that records the edge-specific effects for a `SwitchInt` terminator.
pub trait SwitchIntEdgeEffects<D> {
    /// Calls `apply_edge_effect` for each outgoing edge from a `SwitchInt` terminator and
    /// records the results.
    fn apply(&mut self, apply_edge_effect: impl FnMut(&mut D, SwitchIntTarget));
}

/// List of places that are written to after a successful (non-unwind) return
/// from a `Call` or `InlineAsm`.
pub enum CallReturnPlaces<'a, 'tcx> {
    Call(mir::Place<'tcx>),
    InlineAsm(&'a [mir::InlineAsmOperand<'tcx>]),
}

impl<'tcx> CallReturnPlaces<'_, 'tcx> {
    pub fn for_each(&self, mut f: impl FnMut(mir::Place<'tcx>)) {
        match *self {
            Self::Call(place) => f(place),
            Self::InlineAsm(operands) => {
                for op in operands {
                    match *op {
                        mir::InlineAsmOperand::Out { place: Some(place), .. }
                        | mir::InlineAsmOperand::InOut { out_place: Some(place), .. } => f(place),
                        _ => {}
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests;
