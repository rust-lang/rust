use rustc_data_structures::captures::Captures;
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::graph::WithNumNodes;
use rustc_index::bit_set::BitSet;
use rustc_index::IndexVec;
use rustc_middle::mir::coverage::*;

use super::graph::{BasicCoverageBlock, CoverageGraph, TraverseCoverageGraphWithLoops};

use std::fmt::{self, Debug};

/// The coverage counter or counter expression associated with a particular
/// BCB node or BCB edge.
#[derive(Clone, Copy)]
pub(super) enum BcbCounter {
    Counter { id: CounterId },
    Expression { id: ExpressionId },
}

impl BcbCounter {
    fn is_expression(&self) -> bool {
        matches!(self, Self::Expression { .. })
    }

    pub(super) fn as_term(&self) -> CovTerm {
        match *self {
            BcbCounter::Counter { id, .. } => CovTerm::Counter(id),
            BcbCounter::Expression { id, .. } => CovTerm::Expression(id),
        }
    }
}

impl Debug for BcbCounter {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Counter { id, .. } => write!(fmt, "Counter({:?})", id.index()),
            Self::Expression { id } => write!(fmt, "Expression({:?})", id.index()),
        }
    }
}

#[derive(Debug)]
pub(super) enum CounterIncrementSite {
    Node { bcb: BasicCoverageBlock },
    Edge { from_bcb: BasicCoverageBlock, to_bcb: BasicCoverageBlock },
}

/// Generates and stores coverage counter and coverage expression information
/// associated with nodes/edges in the BCB graph.
pub(super) struct CoverageCounters {
    /// List of places where a counter-increment statement should be injected
    /// into MIR, each with its corresponding counter ID.
    counter_increment_sites: IndexVec<CounterId, CounterIncrementSite>,

    /// Coverage counters/expressions that are associated with individual BCBs.
    bcb_counters: IndexVec<BasicCoverageBlock, Option<BcbCounter>>,
    /// Coverage counters/expressions that are associated with the control-flow
    /// edge between two BCBs.
    ///
    /// We currently don't iterate over this map, but if we do in the future,
    /// switch it back to `FxIndexMap` to avoid query stability hazards.
    bcb_edge_counters: FxHashMap<(BasicCoverageBlock, BasicCoverageBlock), BcbCounter>,
    /// Tracks which BCBs have a counter associated with some incoming edge.
    /// Only used by assertions, to verify that BCBs with incoming edge
    /// counters do not have their own physical counters (expressions are allowed).
    bcb_has_incoming_edge_counters: BitSet<BasicCoverageBlock>,
    /// Table of expression data, associating each expression ID with its
    /// corresponding operator (+ or -) and its LHS/RHS operands.
    expressions: IndexVec<ExpressionId, Expression>,
}

impl CoverageCounters {
    /// Makes [`BcbCounter`] `Counter`s and `Expressions` for the `BasicCoverageBlock`s directly or
    /// indirectly associated with coverage spans, and accumulates additional `Expression`s
    /// representing intermediate values.
    pub(super) fn make_bcb_counters(
        basic_coverage_blocks: &CoverageGraph,
        bcb_has_coverage_spans: impl Fn(BasicCoverageBlock) -> bool,
    ) -> Self {
        let num_bcbs = basic_coverage_blocks.num_nodes();

        let mut this = Self {
            counter_increment_sites: IndexVec::new(),
            bcb_counters: IndexVec::from_elem_n(None, num_bcbs),
            bcb_edge_counters: FxHashMap::default(),
            bcb_has_incoming_edge_counters: BitSet::new_empty(num_bcbs),
            expressions: IndexVec::new(),
        };

        MakeBcbCounters::new(&mut this, basic_coverage_blocks)
            .make_bcb_counters(bcb_has_coverage_spans);

        this
    }

    fn make_counter(&mut self, site: CounterIncrementSite) -> BcbCounter {
        let id = self.counter_increment_sites.push(site);
        BcbCounter::Counter { id }
    }

    fn make_expression(&mut self, lhs: BcbCounter, op: Op, rhs: BcbCounter) -> BcbCounter {
        let expression = Expression { lhs: lhs.as_term(), op, rhs: rhs.as_term() };
        let id = self.expressions.push(expression);
        BcbCounter::Expression { id }
    }

    /// Variant of `make_expression` that makes `lhs` optional and assumes [`Op::Add`].
    ///
    /// This is useful when using [`Iterator::fold`] to build an arbitrary-length sum.
    fn make_sum_expression(&mut self, lhs: Option<BcbCounter>, rhs: BcbCounter) -> BcbCounter {
        let Some(lhs) = lhs else { return rhs };
        self.make_expression(lhs, Op::Add, rhs)
    }

    pub(super) fn num_counters(&self) -> usize {
        self.counter_increment_sites.len()
    }

    #[cfg(test)]
    pub(super) fn num_expressions(&self) -> usize {
        self.expressions.len()
    }

    fn set_bcb_counter(&mut self, bcb: BasicCoverageBlock, counter_kind: BcbCounter) -> BcbCounter {
        assert!(
            // If the BCB has an edge counter (to be injected into a new `BasicBlock`), it can also
            // have an expression (to be injected into an existing `BasicBlock` represented by this
            // `BasicCoverageBlock`).
            counter_kind.is_expression() || !self.bcb_has_incoming_edge_counters.contains(bcb),
            "attempt to add a `Counter` to a BCB target with existing incoming edge counters"
        );

        if let Some(replaced) = self.bcb_counters[bcb].replace(counter_kind) {
            bug!(
                "attempt to set a BasicCoverageBlock coverage counter more than once; \
                {bcb:?} already had counter {replaced:?}",
            );
        } else {
            counter_kind
        }
    }

    fn set_bcb_edge_counter(
        &mut self,
        from_bcb: BasicCoverageBlock,
        to_bcb: BasicCoverageBlock,
        counter_kind: BcbCounter,
    ) -> BcbCounter {
        // If the BCB has an edge counter (to be injected into a new `BasicBlock`), it can also
        // have an expression (to be injected into an existing `BasicBlock` represented by this
        // `BasicCoverageBlock`).
        if let Some(node_counter) = self.bcb_counter(to_bcb)
            && !node_counter.is_expression()
        {
            bug!(
                "attempt to add an incoming edge counter from {from_bcb:?} \
                when the target BCB already has {node_counter:?}"
            );
        }

        self.bcb_has_incoming_edge_counters.insert(to_bcb);
        if let Some(replaced) = self.bcb_edge_counters.insert((from_bcb, to_bcb), counter_kind) {
            bug!(
                "attempt to set an edge counter more than once; from_bcb: \
                {from_bcb:?} already had counter {replaced:?}",
            );
        } else {
            counter_kind
        }
    }

    pub(super) fn bcb_counter(&self, bcb: BasicCoverageBlock) -> Option<BcbCounter> {
        self.bcb_counters[bcb]
    }

    /// Returns an iterator over all the nodes/edges in the coverage graph that
    /// should have a counter-increment statement injected into MIR, along with
    /// each site's corresponding counter ID.
    pub(super) fn counter_increment_sites(
        &self,
    ) -> impl Iterator<Item = (CounterId, &CounterIncrementSite)> {
        self.counter_increment_sites.iter_enumerated()
    }

    /// Returns an iterator over the subset of BCB nodes that have been associated
    /// with a counter *expression*, along with the ID of that expression.
    pub(super) fn bcb_nodes_with_coverage_expressions(
        &self,
    ) -> impl Iterator<Item = (BasicCoverageBlock, ExpressionId)> + Captures<'_> {
        self.bcb_counters.iter_enumerated().filter_map(|(bcb, &counter_kind)| match counter_kind {
            // Yield the BCB along with its associated expression ID.
            Some(BcbCounter::Expression { id }) => Some((bcb, id)),
            // This BCB is associated with a counter or nothing, so skip it.
            Some(BcbCounter::Counter { .. }) | None => None,
        })
    }

    pub(super) fn into_expressions(self) -> IndexVec<ExpressionId, Expression> {
        self.expressions
    }
}

/// Traverse the `CoverageGraph` and add either a `Counter` or `Expression` to every BCB, to be
/// injected with coverage spans. `Expressions` have no runtime overhead, so if a viable expression
/// (adding or subtracting two other counters or expressions) can compute the same result as an
/// embedded counter, an `Expression` should be used.
struct MakeBcbCounters<'a> {
    coverage_counters: &'a mut CoverageCounters,
    basic_coverage_blocks: &'a CoverageGraph,
}

impl<'a> MakeBcbCounters<'a> {
    fn new(
        coverage_counters: &'a mut CoverageCounters,
        basic_coverage_blocks: &'a CoverageGraph,
    ) -> Self {
        Self { coverage_counters, basic_coverage_blocks }
    }

    /// If two `BasicCoverageBlock`s branch from another `BasicCoverageBlock`, one of the branches
    /// can be counted by `Expression` by subtracting the other branch from the branching
    /// block. Otherwise, the `BasicCoverageBlock` executed the least should have the `Counter`.
    /// One way to predict which branch executes the least is by considering loops. A loop is exited
    /// at a branch, so the branch that jumps to a `BasicCoverageBlock` outside the loop is almost
    /// always executed less than the branch that does not exit the loop.
    fn make_bcb_counters(&mut self, bcb_has_coverage_spans: impl Fn(BasicCoverageBlock) -> bool) {
        debug!("make_bcb_counters(): adding a counter or expression to each BasicCoverageBlock");

        // Walk the `CoverageGraph`. For each `BasicCoverageBlock` node with an associated
        // coverage span, add a counter. If the `BasicCoverageBlock` branches, add a counter or
        // expression to each branch `BasicCoverageBlock` (if the branch BCB has only one incoming
        // edge) or edge from the branching BCB to the branch BCB (if the branch BCB has multiple
        // incoming edges).
        //
        // The `TraverseCoverageGraphWithLoops` traversal ensures that, when a loop is encountered,
        // all `BasicCoverageBlock` nodes in the loop are visited before visiting any node outside
        // the loop. The `traversal` state includes a `context_stack`, providing a way to know if
        // the current BCB is in one or more nested loops or not.
        let mut traversal = TraverseCoverageGraphWithLoops::new(self.basic_coverage_blocks);
        while let Some(bcb) = traversal.next() {
            if bcb_has_coverage_spans(bcb) {
                debug!("{:?} has at least one coverage span. Get or make its counter", bcb);
                self.make_node_and_branch_counters(&traversal, bcb);
            } else {
                debug!(
                    "{:?} does not have any coverage spans. A counter will only be added if \
                    and when a covered BCB has an expression dependency.",
                    bcb,
                );
            }
        }

        assert!(
            traversal.is_complete(),
            "`TraverseCoverageGraphWithLoops` missed some `BasicCoverageBlock`s: {:?}",
            traversal.unvisited(),
        );
    }

    fn make_node_and_branch_counters(
        &mut self,
        traversal: &TraverseCoverageGraphWithLoops<'_>,
        from_bcb: BasicCoverageBlock,
    ) {
        // First, ensure that this node has a counter of some kind.
        // We might also use its term later to compute one of the branch counters.
        let from_bcb_operand = self.get_or_make_counter_operand(from_bcb);

        let branch_target_bcbs = self.basic_coverage_blocks.successors[from_bcb].as_slice();

        // If this node doesn't have multiple out-edges, or all of its out-edges
        // already have counters, then we don't need to create edge counters.
        let needs_branch_counters = branch_target_bcbs.len() > 1
            && branch_target_bcbs
                .iter()
                .any(|&to_bcb| self.branch_has_no_counter(from_bcb, to_bcb));
        if !needs_branch_counters {
            return;
        }

        debug!(
            "{from_bcb:?} has some branch(es) without counters:\n  {}",
            branch_target_bcbs
                .iter()
                .map(|&to_bcb| {
                    format!("{from_bcb:?}->{to_bcb:?}: {:?}", self.branch_counter(from_bcb, to_bcb))
                })
                .collect::<Vec<_>>()
                .join("\n  "),
        );

        // Of the branch edges that don't have counters yet, one can be given an expression
        // (computed from the other edges) instead of a dedicated counter.
        let expression_to_bcb = self.choose_preferred_expression_branch(traversal, from_bcb);

        // For each branch arm other than the one that was chosen to get an expression,
        // ensure that it has a counter (existing counter/expression or a new counter),
        // and accumulate the corresponding terms into a single sum term.
        let sum_of_all_other_branches: BcbCounter = {
            let _span = debug_span!("sum_of_all_other_branches", ?expression_to_bcb).entered();
            branch_target_bcbs
                .iter()
                .copied()
                // Skip the chosen branch, since we'll calculate it from the other branches.
                .filter(|&to_bcb| to_bcb != expression_to_bcb)
                .fold(None, |accum, to_bcb| {
                    let _span = debug_span!("to_bcb", ?accum, ?to_bcb).entered();
                    let branch_counter = self.get_or_make_edge_counter_operand(from_bcb, to_bcb);
                    Some(self.coverage_counters.make_sum_expression(accum, branch_counter))
                })
                .expect("there must be at least one other branch")
        };

        // For the branch that was chosen to get an expression, create that expression
        // by taking the count of the node we're branching from, and subtracting the
        // sum of all the other branches.
        debug!(
            "Making an expression for the selected expression_branch: \
            {expression_to_bcb:?} (expression_branch predecessors: {:?})",
            self.bcb_predecessors(expression_to_bcb),
        );
        let expression = self.coverage_counters.make_expression(
            from_bcb_operand,
            Op::Subtract,
            sum_of_all_other_branches,
        );
        debug!("{expression_to_bcb:?} gets an expression: {expression:?}");
        if self.basic_coverage_blocks.bcb_has_multiple_in_edges(expression_to_bcb) {
            self.coverage_counters.set_bcb_edge_counter(from_bcb, expression_to_bcb, expression);
        } else {
            self.coverage_counters.set_bcb_counter(expression_to_bcb, expression);
        }
    }

    #[instrument(level = "debug", skip(self))]
    fn get_or_make_counter_operand(&mut self, bcb: BasicCoverageBlock) -> BcbCounter {
        // If the BCB already has a counter, return it.
        if let Some(counter_kind) = self.coverage_counters.bcb_counters[bcb] {
            debug!("{bcb:?} already has a counter: {counter_kind:?}");
            return counter_kind;
        }

        // A BCB with only one incoming edge gets a simple `Counter` (via `make_counter()`).
        // Also, a BCB that loops back to itself gets a simple `Counter`. This may indicate the
        // program results in a tight infinite loop, but it should still compile.
        let one_path_to_target = !self.basic_coverage_blocks.bcb_has_multiple_in_edges(bcb);
        if one_path_to_target || self.bcb_predecessors(bcb).contains(&bcb) {
            let counter_kind =
                self.coverage_counters.make_counter(CounterIncrementSite::Node { bcb });
            if one_path_to_target {
                debug!("{bcb:?} gets a new counter: {counter_kind:?}");
            } else {
                debug!(
                    "{bcb:?} has itself as its own predecessor. It can't be part of its own \
                    Expression sum, so it will get its own new counter: {counter_kind:?}. \
                    (Note, the compiled code will generate an infinite loop.)",
                );
            }
            return self.coverage_counters.set_bcb_counter(bcb, counter_kind);
        }

        // A BCB with multiple incoming edges can compute its count by ensuring that counters
        // exist for each of those edges, and then adding them up to get a total count.
        let sum_of_in_edges: BcbCounter = {
            let _span = debug_span!("sum_of_in_edges", ?bcb).entered();
            // We avoid calling `self.bcb_predecessors` here so that we can
            // call methods on `&mut self` inside the fold.
            self.basic_coverage_blocks.predecessors[bcb]
                .iter()
                .copied()
                .fold(None, |accum, from_bcb| {
                    let _span = debug_span!("from_bcb", ?accum, ?from_bcb).entered();
                    let edge_counter = self.get_or_make_edge_counter_operand(from_bcb, bcb);
                    Some(self.coverage_counters.make_sum_expression(accum, edge_counter))
                })
                .expect("there must be at least one in-edge")
        };

        debug!("{bcb:?} gets a new counter (sum of predecessor counters): {sum_of_in_edges:?}");
        self.coverage_counters.set_bcb_counter(bcb, sum_of_in_edges)
    }

    #[instrument(level = "debug", skip(self))]
    fn get_or_make_edge_counter_operand(
        &mut self,
        from_bcb: BasicCoverageBlock,
        to_bcb: BasicCoverageBlock,
    ) -> BcbCounter {
        // If the target BCB has only one in-edge (i.e. this one), then create
        // a node counter instead, since it will have the same value.
        if !self.basic_coverage_blocks.bcb_has_multiple_in_edges(to_bcb) {
            assert_eq!([from_bcb].as_slice(), self.basic_coverage_blocks.predecessors[to_bcb]);
            return self.get_or_make_counter_operand(to_bcb);
        }

        // If the source BCB has only one successor (assumed to be the given target), an edge
        // counter is unnecessary. Just get or make a counter for the source BCB.
        if self.bcb_successors(from_bcb).len() == 1 {
            return self.get_or_make_counter_operand(from_bcb);
        }

        // If the edge already has a counter, return it.
        if let Some(&counter_kind) =
            self.coverage_counters.bcb_edge_counters.get(&(from_bcb, to_bcb))
        {
            debug!("Edge {from_bcb:?}->{to_bcb:?} already has a counter: {counter_kind:?}");
            return counter_kind;
        }

        // Make a new counter to count this edge.
        let counter_kind =
            self.coverage_counters.make_counter(CounterIncrementSite::Edge { from_bcb, to_bcb });
        debug!("Edge {from_bcb:?}->{to_bcb:?} gets a new counter: {counter_kind:?}");
        self.coverage_counters.set_bcb_edge_counter(from_bcb, to_bcb, counter_kind)
    }

    /// Select a branch for the expression, either the recommended `reloop_branch`, or if none was
    /// found, select any branch.
    fn choose_preferred_expression_branch(
        &self,
        traversal: &TraverseCoverageGraphWithLoops<'_>,
        from_bcb: BasicCoverageBlock,
    ) -> BasicCoverageBlock {
        let good_reloop_branch = self.find_good_reloop_branch(traversal, from_bcb);
        if let Some(reloop_target) = good_reloop_branch {
            assert!(self.branch_has_no_counter(from_bcb, reloop_target));
            debug!("Selecting reloop target {reloop_target:?} to get an expression");
            reloop_target
        } else {
            let &branch_without_counter = self
                .bcb_successors(from_bcb)
                .iter()
                .find(|&&to_bcb| self.branch_has_no_counter(from_bcb, to_bcb))
                .expect(
                    "needs_branch_counters was `true` so there should be at least one \
                    branch",
                );
            debug!(
                "Selecting any branch={:?} that still needs a counter, to get the \
                `Expression` because there was no `reloop_branch`, or it already had a \
                counter",
                branch_without_counter
            );
            branch_without_counter
        }
    }

    /// Tries to find a branch that leads back to the top of a loop, and that
    /// doesn't already have a counter. Such branches are good candidates to
    /// be given an expression (instead of a physical counter), because they
    /// will tend to be executed more times than a loop-exit branch.
    fn find_good_reloop_branch(
        &self,
        traversal: &TraverseCoverageGraphWithLoops<'_>,
        from_bcb: BasicCoverageBlock,
    ) -> Option<BasicCoverageBlock> {
        let branch_target_bcbs = self.bcb_successors(from_bcb);

        // Consider each loop on the current traversal context stack, top-down.
        for reloop_bcbs in traversal.reloop_bcbs_per_loop() {
            let mut all_branches_exit_this_loop = true;

            // Try to find a branch that doesn't exit this loop and doesn't
            // already have a counter.
            for &branch_target_bcb in branch_target_bcbs {
                // A branch is a reloop branch if it dominates any BCB that has
                // an edge back to the loop header. (Other branches are exits.)
                let is_reloop_branch = reloop_bcbs.iter().any(|&reloop_bcb| {
                    self.basic_coverage_blocks.dominates(branch_target_bcb, reloop_bcb)
                });

                if is_reloop_branch {
                    all_branches_exit_this_loop = false;
                    if self.branch_has_no_counter(from_bcb, branch_target_bcb) {
                        // We found a good branch to be given an expression.
                        return Some(branch_target_bcb);
                    }
                    // Keep looking for another reloop branch without a counter.
                } else {
                    // This branch exits the loop.
                }
            }

            if !all_branches_exit_this_loop {
                // We found one or more reloop branches, but all of them already
                // have counters. Let the caller choose one of the exit branches.
                debug!("All reloop branches had counters; skip checking the other loops");
                return None;
            }

            // All of the branches exit this loop, so keep looking for a good
            // reloop branch for one of the outer loops.
        }

        None
    }

    #[inline]
    fn bcb_predecessors(&self, bcb: BasicCoverageBlock) -> &[BasicCoverageBlock] {
        &self.basic_coverage_blocks.predecessors[bcb]
    }

    #[inline]
    fn bcb_successors(&self, bcb: BasicCoverageBlock) -> &[BasicCoverageBlock] {
        &self.basic_coverage_blocks.successors[bcb]
    }

    #[inline]
    fn branch_has_no_counter(
        &self,
        from_bcb: BasicCoverageBlock,
        to_bcb: BasicCoverageBlock,
    ) -> bool {
        self.branch_counter(from_bcb, to_bcb).is_none()
    }

    fn branch_counter(
        &self,
        from_bcb: BasicCoverageBlock,
        to_bcb: BasicCoverageBlock,
    ) -> Option<&BcbCounter> {
        if self.basic_coverage_blocks.bcb_has_multiple_in_edges(to_bcb) {
            self.coverage_counters.bcb_edge_counters.get(&(from_bcb, to_bcb))
        } else {
            self.coverage_counters.bcb_counters[to_bcb].as_ref()
        }
    }
}
