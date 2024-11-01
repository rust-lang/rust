//! A framework that can express both [gen-kill] and generic dataflow problems.
//!
//! To use this framework, implement the [`Analysis`] trait. There used to be a `GenKillAnalysis`
//! alternative trait for gen-kill analyses that would pre-compute the transfer function for each
//! block. It was intended as an optimization, but it ended up not being any faster than
//! `Analysis`.
//!
//! The `impls` module contains several examples of dataflow analyses.
//!
//! Then call `iterate_to_fixpoint` on your type that impls `Analysis` to get a `Results`. From
//! there, you can use a `ResultsCursor` to inspect the fixpoint solution to your dataflow problem,
//! or implement the `ResultsVisitor` interface and use `visit_results`. The following example uses
//! the `ResultsCursor` approach.
//!
//! ```ignore (cross-crate-imports)
//! use rustc_const_eval::dataflow::Analysis; // Makes `iterate_to_fixpoint` available.
//!
//! fn do_my_analysis(tcx: TyCtxt<'tcx>, body: &mir::Body<'tcx>) {
//!     let analysis = MyAnalysis::new()
//!         .iterate_to_fixpoint(tcx, body, None)
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

use rustc_data_structures::work_queue::WorkQueue;
use rustc_index::bit_set::{BitSet, ChunkedBitSet, HybridBitSet};
use rustc_index::{Idx, IndexVec};
use rustc_middle::bug;
use rustc_middle::mir::{self, BasicBlock, CallReturnPlaces, Location, TerminatorEdges, traversal};
use rustc_middle::ty::TyCtxt;
use tracing::error;

use self::results::write_graphviz_results;
use super::fmt::DebugWithContext;

mod cursor;
mod direction;
pub mod fmt;
pub mod graphviz;
pub mod lattice;
mod results;
mod visitor;

pub use self::cursor::ResultsCursor;
pub use self::direction::{Backward, Direction, Forward};
pub use self::lattice::{JoinSemiLattice, MaybeReachable};
pub use self::results::Results;
pub use self::visitor::{ResultsVisitable, ResultsVisitor, visit_results};

/// Analysis domains are all bitsets of various kinds. This trait holds
/// operations needed by all of them.
pub trait BitSetExt<T> {
    fn contains(&self, elem: T) -> bool;
    fn union(&mut self, other: &HybridBitSet<T>);
    fn subtract(&mut self, other: &HybridBitSet<T>);
}

impl<T: Idx> BitSetExt<T> for BitSet<T> {
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

/// A dataflow problem with an arbitrarily complex transfer function.
///
/// This trait specifies the lattice on which this analysis operates (the domain), its
/// initial value at the entry point of each basic block, and various operations.
///
/// # Convergence
///
/// When implementing this trait it's possible to choose a transfer function such that the analysis
/// does not reach fixpoint. To guarantee convergence, your transfer functions must maintain the
/// following invariant:
///
/// > If the dataflow state **before** some point in the program changes to be greater
/// than the prior state **before** that point, the dataflow state **after** that point must
/// also change to be greater than the prior state **after** that point.
///
/// This invariant guarantees that the dataflow state at a given point in the program increases
/// monotonically until fixpoint is reached. Note that this monotonicity requirement only applies
/// to the same point in the program at different points in time. The dataflow state at a given
/// point in the program may or may not be greater than the state at any preceding point.
pub trait Analysis<'tcx> {
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
    // `resume`). It's not obvious how to handle `yield` points in coroutines, however.
    fn initialize_start_block(&self, body: &mir::Body<'tcx>, state: &mut Self::Domain);

    /// Updates the current dataflow state with the effect of evaluating a statement.
    fn apply_statement_effect(
        &mut self,
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
        &mut self,
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
    fn apply_terminator_effect<'mir>(
        &mut self,
        _state: &mut Self::Domain,
        terminator: &'mir mir::Terminator<'tcx>,
        _location: Location,
    ) -> TerminatorEdges<'mir, 'tcx> {
        terminator.edges()
    }

    /// Updates the current dataflow state with an effect that occurs immediately *before* the
    /// given terminator.
    ///
    /// This method is useful if the consumer of the results of this analysis needs only to observe
    /// *part* of the effect of a terminator (e.g. for two-phase borrows). As a general rule,
    /// analyses should not implement this without also implementing `apply_terminator_effect`.
    fn apply_before_terminator_effect(
        &mut self,
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
        &mut self,
        _state: &mut Self::Domain,
        _block: BasicBlock,
        _return_places: CallReturnPlaces<'_, 'tcx>,
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
        &mut self,
        _block: BasicBlock,
        _discr: &mir::Operand<'tcx>,
        _apply_edge_effects: &mut impl SwitchIntEdgeEffects<Self::Domain>,
    ) {
    }

    /* Extension methods */

    /// Finds the fixpoint for this dataflow problem.
    ///
    /// You shouldn't need to override this. Its purpose is to enable method chaining like so:
    ///
    /// ```ignore (cross-crate-imports)
    /// let results = MyAnalysis::new(tcx, body)
    ///     .iterate_to_fixpoint(tcx, body, None)
    ///     .into_results_cursor(body);
    /// ```
    /// You can optionally add a `pass_name` to the graphviz output for this particular run of a
    /// dataflow analysis. Some analyses are run multiple times in the compilation pipeline.
    /// Without a `pass_name` to differentiates them, only the results for the latest run will be
    /// saved.
    fn iterate_to_fixpoint<'mir>(
        mut self,
        tcx: TyCtxt<'tcx>,
        body: &'mir mir::Body<'tcx>,
        pass_name: Option<&'static str>,
    ) -> Results<'tcx, Self>
    where
        Self: Sized,
        Self::Domain: DebugWithContext<Self>,
    {
        let mut entry_sets =
            IndexVec::from_fn_n(|_| self.bottom_value(body), body.basic_blocks.len());
        self.initialize_start_block(body, &mut entry_sets[mir::START_BLOCK]);

        if Self::Direction::IS_BACKWARD && entry_sets[mir::START_BLOCK] != self.bottom_value(body) {
            bug!("`initialize_start_block` is not yet supported for backward dataflow analyses");
        }

        let mut dirty_queue: WorkQueue<BasicBlock> = WorkQueue::with_none(body.basic_blocks.len());

        if Self::Direction::IS_FORWARD {
            for (bb, _) in traversal::reverse_postorder(body) {
                dirty_queue.insert(bb);
            }
        } else {
            // Reverse post-order on the reverse CFG may generate a better iteration order for
            // backward dataflow analyses, but probably not enough to matter.
            for (bb, _) in traversal::postorder(body) {
                dirty_queue.insert(bb);
            }
        }

        // `state` is not actually used between iterations;
        // this is just an optimization to avoid reallocating
        // every iteration.
        let mut state = self.bottom_value(body);
        while let Some(bb) = dirty_queue.pop() {
            let bb_data = &body[bb];

            // Set the state to the entry state of the block.
            // This is equivalent to `state = entry_sets[bb].clone()`,
            // but it saves an allocation, thus improving compile times.
            state.clone_from(&entry_sets[bb]);

            // Apply the block transfer function, using the cached one if it exists.
            let edges = Self::Direction::apply_effects_in_block(&mut self, &mut state, bb, bb_data);

            Self::Direction::join_state_into_successors_of(
                &mut self,
                body,
                &mut state,
                bb,
                edges,
                |target: BasicBlock, state: &Self::Domain| {
                    let set_changed = entry_sets[target].join(state);
                    if set_changed {
                        dirty_queue.insert(target);
                    }
                },
            );
        }

        let results = Results { analysis: self, entry_sets };

        if tcx.sess.opts.unstable_opts.dump_mir_dataflow {
            let (res, results) = write_graphviz_results(tcx, body, results, pass_name);
            if let Err(e) = res {
                error!("Failed to write graphviz dataflow results: {}", e);
            }
            results
        } else {
            results
        }
    }
}

/// The legal operations for a transfer function in a gen/kill problem.
pub trait GenKill<T> {
    /// Inserts `elem` into the state vector.
    fn gen_(&mut self, elem: T);

    /// Removes `elem` from the state vector.
    fn kill(&mut self, elem: T);

    /// Calls `gen` for each element in `elems`.
    fn gen_all(&mut self, elems: impl IntoIterator<Item = T>) {
        for elem in elems {
            self.gen_(elem);
        }
    }

    /// Calls `kill` for each element in `elems`.
    fn kill_all(&mut self, elems: impl IntoIterator<Item = T>) {
        for elem in elems {
            self.kill(elem);
        }
    }
}

impl<T: Idx> GenKill<T> for BitSet<T> {
    fn gen_(&mut self, elem: T) {
        self.insert(elem);
    }

    fn kill(&mut self, elem: T) {
        self.remove(elem);
    }
}

impl<T: Idx> GenKill<T> for ChunkedBitSet<T> {
    fn gen_(&mut self, elem: T) {
        self.insert(elem);
    }

    fn kill(&mut self, elem: T) {
        self.remove(elem);
    }
}

impl<T, S: GenKill<T>> GenKill<T> for MaybeReachable<S> {
    fn gen_(&mut self, elem: T) {
        match self {
            // If the state is not reachable, adding an element does nothing.
            MaybeReachable::Unreachable => {}
            MaybeReachable::Reachable(set) => set.gen_(elem),
        }
    }

    fn kill(&mut self, elem: T) {
        match self {
            // If the state is not reachable, killing an element does nothing.
            MaybeReachable::Unreachable => {}
            MaybeReachable::Reachable(set) => set.kill(elem),
        }
    }
}

impl<T: Idx> GenKill<T> for lattice::Dual<BitSet<T>> {
    fn gen_(&mut self, elem: T) {
        self.0.insert(elem);
    }

    fn kill(&mut self, elem: T) {
        self.0.remove(elem);
    }
}

// NOTE: DO NOT CHANGE VARIANT ORDER. The derived `Ord` impls rely on the current order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Effect {
    /// The "before" effect (e.g., `apply_before_statement_effect`) for a statement (or
    /// terminator).
    Before,

    /// The "primary" effect (e.g., `apply_statement_effect`) for a statement (or terminator).
    Primary,
}

impl Effect {
    const fn at_index(self, statement_index: usize) -> EffectIndex {
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

#[cfg(test)]
mod tests;
