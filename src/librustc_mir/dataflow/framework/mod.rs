//! A framework that can express both [gen-kill] and generic dataflow problems.
//!
//! To actually use this framework, you must implement either the `Analysis` or the
//! `GenKillAnalysis` trait. If your transfer function can be expressed with only gen/kill
//! operations, prefer `GenKillAnalysis` since it will run faster while iterating to fixpoint. The
//! `impls` module contains several examples of gen/kill dataflow analyses.
//!
//! Create an `Engine` for your analysis using the `into_engine` method on the `Analysis` trait,
//! then call `iterate_to_fixpoint`. From there, you can use a `ResultsCursor` to inspect the
//! fixpoint solution to your dataflow problem, or implement the `ResultsVisitor` interface and use
//! `visit_results`. The following example uses the `ResultsCursor` approach.
//!
//! ```ignore(cross-crate-imports)
//! use rustc_mir::dataflow::Analysis; // Makes `into_engine` available.
//!
//! fn do_my_analysis(tcx: TyCtxt<'tcx>, body: &mir::Body<'tcx>, did: DefId) {
//!     let analysis = MyAnalysis::new()
//!         .into_engine(tcx, body, did)
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
use std::io;

use rustc_hir::def_id::DefId;
use rustc_index::bit_set::{BitSet, HybridBitSet};
use rustc_index::vec::Idx;
use rustc_middle::mir::{self, BasicBlock, Location};
use rustc_middle::ty::{self, TyCtxt};
use rustc_target::abi::VariantIdx;

mod cursor;
mod direction;
mod engine;
mod graphviz;
mod visitor;

pub use self::cursor::{ResultsCursor, ResultsRefCursor};
pub use self::direction::{Backward, Direction, Forward};
pub use self::engine::{Engine, Results};
pub use self::visitor::{visit_results, ResultsVisitor};
pub use self::visitor::{BorrowckFlowState, BorrowckResults};

/// Parameterization for the precise form of data flow that is used.
///
/// `BottomValue` determines whether the initial entry set for each basic block is empty or full.
/// This also determines the semantics of the lattice `join` operator used to merge dataflow
/// results, since dataflow works by starting at the bottom and moving monotonically to a fixed
/// point.
///
/// This means, for propagation across the graph, that you either want to start at all-zeroes and
/// then use Union as your merge when propagating, or you start at all-ones and then use Intersect
/// as your merge when propagating.
pub trait BottomValue {
    /// Specifies the initial value for each bit in the entry set for each basic block.
    const BOTTOM_VALUE: bool;

    /// Merges `in_set` into `inout_set`, returning `true` if `inout_set` changed.
    ///
    /// It is almost certainly wrong to override this, since it automatically applies
    /// * `inout_set & in_set` if `BOTTOM_VALUE == true`
    /// * `inout_set | in_set` if `BOTTOM_VALUE == false`
    ///
    /// This means that if a bit is not `BOTTOM_VALUE`, it is propagated into all target blocks.
    /// For clarity, the above statement again from a different perspective:
    /// A bit in the block's entry set is `!BOTTOM_VALUE` if *any* predecessor block's bit value is
    /// `!BOTTOM_VALUE`.
    ///
    /// There are situations where you want the opposite behaviour: propagate only if *all*
    /// predecessor blocks's value is `!BOTTOM_VALUE`.
    /// E.g. if you want to know whether a bit is *definitely* set at a specific location. This
    /// means that all code paths leading to the location must have set the bit, instead of any
    /// code path leading there.
    ///
    /// If you want this kind of "definitely set" analysis, you need to
    /// 1. Invert `BOTTOM_VALUE`
    /// 2. Reset the `entry_set` in `start_block_effect` to `!BOTTOM_VALUE`
    /// 3. Override `join` to do the opposite from what it's doing now.
    #[inline]
    fn join<T: Idx>(&self, inout_set: &mut BitSet<T>, in_set: &BitSet<T>) -> bool {
        if !Self::BOTTOM_VALUE { inout_set.union(in_set) } else { inout_set.intersect(in_set) }
    }
}

/// Define the domain of a dataflow problem.
///
/// This trait specifies the lattice on which this analysis operates. For now, this must be a
/// powerset of values of type `Idx`. The elements of this lattice are represented with a `BitSet`
/// and referred to as the state vector.
///
/// This trait also defines the initial value for the dataflow state upon entry to the
/// `START_BLOCK`, as well as some names used to refer to this analysis when debugging.
pub trait AnalysisDomain<'tcx>: BottomValue {
    /// The type of the elements in the state vector.
    type Idx: Idx;

    /// The direction of this analyis. Either `Forward` or `Backward`.
    type Direction: Direction = Forward;

    /// A descriptive name for this analysis. Used only for debugging.
    ///
    /// This name should be brief and contain no spaces, periods or other characters that are not
    /// suitable as part of a filename.
    const NAME: &'static str;

    /// The size of the state vector.
    fn bits_per_block(&self, body: &mir::Body<'tcx>) -> usize;

    /// Mutates the entry set of the `START_BLOCK` to contain the initial state for dataflow
    /// analysis.
    ///
    /// For backward analyses, initial state besides the bottom value is not yet supported. Trying
    /// to mutate the initial state will result in a panic.
    //
    // FIXME: For backward dataflow analyses, the initial state should be applied to every basic
    // block where control flow could exit the MIR body (e.g., those terminated with `return` or
    // `resume`). It's not obvious how to handle `yield` points in generators, however.
    fn initialize_start_block(&self, body: &mir::Body<'tcx>, state: &mut BitSet<Self::Idx>);

    /// Prints an element in the state vector for debugging.
    fn pretty_print_idx(&self, w: &mut impl io::Write, idx: Self::Idx) -> io::Result<()> {
        write!(w, "{:?}", idx)
    }
}

/// A dataflow problem with an arbitrarily complex transfer function.
pub trait Analysis<'tcx>: AnalysisDomain<'tcx> {
    /// Updates the current dataflow state with the effect of evaluating a statement.
    fn apply_statement_effect(
        &self,
        state: &mut BitSet<Self::Idx>,
        statement: &mir::Statement<'tcx>,
        location: Location,
    );

    /// Updates the current dataflow state with an effect that occurs immediately *before* the
    /// given statement.
    ///
    /// This method is useful if the consumer of the results of this analysis needs only to observe
    /// *part* of the effect of a statement (e.g. for two-phase borrows). As a general rule,
    /// analyses should not implement this without implementing `apply_statement_effect`.
    fn apply_before_statement_effect(
        &self,
        _state: &mut BitSet<Self::Idx>,
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
        state: &mut BitSet<Self::Idx>,
        terminator: &mir::Terminator<'tcx>,
        location: Location,
    );

    /// Updates the current dataflow state with an effect that occurs immediately *before* the
    /// given terminator.
    ///
    /// This method is useful if the consumer of the results of this analysis needs only to observe
    /// *part* of the effect of a terminator (e.g. for two-phase borrows). As a general rule,
    /// analyses should not implement this without implementing `apply_terminator_effect`.
    fn apply_before_terminator_effect(
        &self,
        _state: &mut BitSet<Self::Idx>,
        _terminator: &mir::Terminator<'tcx>,
        _location: Location,
    ) {
    }

    /// Updates the current dataflow state with the effect of a successful return from a `Call`
    /// terminator.
    ///
    /// This is separate from `apply_terminator_effect` to properly track state across unwind
    /// edges.
    fn apply_call_return_effect(
        &self,
        state: &mut BitSet<Self::Idx>,
        block: BasicBlock,
        func: &mir::Operand<'tcx>,
        args: &[mir::Operand<'tcx>],
        return_place: mir::Place<'tcx>,
    );

    /// Updates the current dataflow state with the effect of resuming from a `Yield` terminator.
    ///
    /// This is similar to `apply_call_return_effect` in that it only takes place after the
    /// generator is resumed, not when it is dropped.
    ///
    /// By default, no effects happen.
    fn apply_yield_resume_effect(
        &self,
        _state: &mut BitSet<Self::Idx>,
        _resume_block: BasicBlock,
        _resume_place: mir::Place<'tcx>,
    ) {
    }

    /// Updates the current dataflow state with the effect of taking a particular branch in a
    /// `SwitchInt` terminator.
    ///
    /// Much like `apply_call_return_effect`, this effect is only propagated along a single
    /// outgoing edge from this basic block.
    ///
    /// FIXME: This class of effects is not supported for backward dataflow analyses.
    fn apply_discriminant_switch_effect(
        &self,
        _state: &mut BitSet<Self::Idx>,
        _block: BasicBlock,
        _enum_place: mir::Place<'tcx>,
        _adt: &ty::AdtDef,
        _variant: VariantIdx,
    ) {
    }

    /// Creates an `Engine` to find the fixpoint for this dataflow problem.
    ///
    /// You shouldn't need to override this outside this module, since the combination of the
    /// default impl and the one for all `A: GenKillAnalysis` will do the right thing.
    /// Its purpose is to enable method chaining like so:
    ///
    /// ```ignore(cross-crate-imports)
    /// let results = MyAnalysis::new(tcx, body)
    ///     .into_engine(tcx, body, def_id)
    ///     .iterate_to_fixpoint()
    ///     .into_results_cursor(body);
    /// ```
    fn into_engine(
        self,
        tcx: TyCtxt<'tcx>,
        body: &'mir mir::Body<'tcx>,
        def_id: DefId,
    ) -> Engine<'mir, 'tcx, Self>
    where
        Self: Sized,
    {
        Engine::new_generic(tcx, body, def_id, self)
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

    /// See `Analysis::apply_call_return_effect`.
    fn call_return_effect(
        &self,
        trans: &mut impl GenKill<Self::Idx>,
        block: BasicBlock,
        func: &mir::Operand<'tcx>,
        args: &[mir::Operand<'tcx>],
        return_place: mir::Place<'tcx>,
    );

    /// See `Analysis::apply_yield_resume_effect`.
    fn yield_resume_effect(
        &self,
        _trans: &mut impl GenKill<Self::Idx>,
        _resume_block: BasicBlock,
        _resume_place: mir::Place<'tcx>,
    ) {
    }

    /// See `Analysis::apply_discriminant_switch_effect`.
    fn discriminant_switch_effect(
        &self,
        _state: &mut impl GenKill<Self::Idx>,
        _block: BasicBlock,
        _enum_place: mir::Place<'tcx>,
        _adt: &ty::AdtDef,
        _variant: VariantIdx,
    ) {
    }
}

impl<A> Analysis<'tcx> for A
where
    A: GenKillAnalysis<'tcx>,
{
    fn apply_statement_effect(
        &self,
        state: &mut BitSet<Self::Idx>,
        statement: &mir::Statement<'tcx>,
        location: Location,
    ) {
        self.statement_effect(state, statement, location);
    }

    fn apply_before_statement_effect(
        &self,
        state: &mut BitSet<Self::Idx>,
        statement: &mir::Statement<'tcx>,
        location: Location,
    ) {
        self.before_statement_effect(state, statement, location);
    }

    fn apply_terminator_effect(
        &self,
        state: &mut BitSet<Self::Idx>,
        terminator: &mir::Terminator<'tcx>,
        location: Location,
    ) {
        self.terminator_effect(state, terminator, location);
    }

    fn apply_before_terminator_effect(
        &self,
        state: &mut BitSet<Self::Idx>,
        terminator: &mir::Terminator<'tcx>,
        location: Location,
    ) {
        self.before_terminator_effect(state, terminator, location);
    }

    fn apply_call_return_effect(
        &self,
        state: &mut BitSet<Self::Idx>,
        block: BasicBlock,
        func: &mir::Operand<'tcx>,
        args: &[mir::Operand<'tcx>],
        return_place: mir::Place<'tcx>,
    ) {
        self.call_return_effect(state, block, func, args, return_place);
    }

    fn apply_yield_resume_effect(
        &self,
        state: &mut BitSet<Self::Idx>,
        resume_block: BasicBlock,
        resume_place: mir::Place<'tcx>,
    ) {
        self.yield_resume_effect(state, resume_block, resume_place);
    }

    fn apply_discriminant_switch_effect(
        &self,
        state: &mut BitSet<Self::Idx>,
        block: BasicBlock,
        enum_place: mir::Place<'tcx>,
        adt: &ty::AdtDef,
        variant: VariantIdx,
    ) {
        self.discriminant_switch_effect(state, block, enum_place, adt, variant);
    }

    fn into_engine(
        self,
        tcx: TyCtxt<'tcx>,
        body: &'mir mir::Body<'tcx>,
        def_id: DefId,
    ) -> Engine<'mir, 'tcx, Self>
    where
        Self: Sized,
    {
        Engine::new_gen_kill(tcx, body, def_id, self)
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
pub struct GenKillSet<T: Idx> {
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

    /// Applies this transfer function to the given state vector.
    pub fn apply(&self, state: &mut BitSet<T>) {
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

    /// Returns `true` if the effect at `self` should be applied eariler than the effect at `other`
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

#[cfg(test)]
mod tests;
