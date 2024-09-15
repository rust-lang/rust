use std::fmt::{self, Debug};

use rustc_data_structures::captures::Captures;
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::graph::DirectedGraph;
use rustc_index::IndexVec;
use rustc_middle::bug;
use rustc_middle::mir::coverage::{CounterId, CovTerm, Expression, ExpressionId, Op};
use tracing::{debug, debug_span, instrument};

use crate::coverage::graph::{BasicCoverageBlock, CoverageGraph, TraverseCoverageGraphWithLoops};

/// The coverage counter or counter expression associated with a particular
/// BCB node or BCB edge.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum BcbCounter {
    Counter { id: CounterId },
    Expression { id: ExpressionId },
}

impl BcbCounter {
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct BcbExpression {
    lhs: BcbCounter,
    op: Op,
    rhs: BcbCounter,
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

    /// Table of expression data, associating each expression ID with its
    /// corresponding operator (+ or -) and its LHS/RHS operands.
    expressions: IndexVec<ExpressionId, BcbExpression>,
    /// Remember expressions that have already been created (or simplified),
    /// so that we don't create unnecessary duplicates.
    expressions_memo: FxHashMap<BcbExpression, BcbCounter>,
}

impl CoverageCounters {
    /// Ensures that each BCB node needing a counter has one, by creating physical
    /// counters or counter expressions for nodes and edges as required.
    pub(super) fn make_bcb_counters(
        basic_coverage_blocks: &CoverageGraph,
        bcb_needs_counter: impl Fn(BasicCoverageBlock) -> bool,
    ) -> Self {
        let num_bcbs = basic_coverage_blocks.num_nodes();

        let mut this = Self {
            counter_increment_sites: IndexVec::new(),
            bcb_counters: IndexVec::from_elem_n(None, num_bcbs),
            bcb_edge_counters: FxHashMap::default(),
            expressions: IndexVec::new(),
            expressions_memo: FxHashMap::default(),
        };

        MakeBcbCounters::new(&mut this, basic_coverage_blocks).make_bcb_counters(bcb_needs_counter);

        this
    }

    /// Shared helper used by [`Self::make_phys_node_counter`] and
    /// [`Self::make_phys_edge_counter`]. Don't call this directly.
    fn make_counter_inner(&mut self, site: CounterIncrementSite) -> BcbCounter {
        let id = self.counter_increment_sites.push(site);
        BcbCounter::Counter { id }
    }

    /// Creates a new physical counter attached a BCB node.
    /// The node must not already have a counter.
    fn make_phys_node_counter(&mut self, bcb: BasicCoverageBlock) -> BcbCounter {
        let counter = self.make_counter_inner(CounterIncrementSite::Node { bcb });
        debug!(?bcb, ?counter, "node gets a physical counter");
        self.set_bcb_counter(bcb, counter)
    }

    /// Creates a new physical counter attached to a BCB edge.
    /// The edge must not already have a counter.
    fn make_phys_edge_counter(
        &mut self,
        from_bcb: BasicCoverageBlock,
        to_bcb: BasicCoverageBlock,
    ) -> BcbCounter {
        let counter = self.make_counter_inner(CounterIncrementSite::Edge { from_bcb, to_bcb });
        debug!(?from_bcb, ?to_bcb, ?counter, "edge gets a physical counter");
        self.set_bcb_edge_counter(from_bcb, to_bcb, counter)
    }

    fn make_expression(&mut self, lhs: BcbCounter, op: Op, rhs: BcbCounter) -> BcbCounter {
        let new_expr = BcbExpression { lhs, op, rhs };
        *self
            .expressions_memo
            .entry(new_expr)
            .or_insert_with(|| Self::make_expression_inner(&mut self.expressions, new_expr))
    }

    /// This is an associated function so that we can call it while borrowing
    /// `&mut self.expressions_memo`.
    fn make_expression_inner(
        expressions: &mut IndexVec<ExpressionId, BcbExpression>,
        new_expr: BcbExpression,
    ) -> BcbCounter {
        // Simplify expressions using basic algebra.
        //
        // Some of these cases might not actually occur in practice, depending
        // on the details of how the instrumentor builds expressions.
        let BcbExpression { lhs, op, rhs } = new_expr;

        if let BcbCounter::Expression { id } = lhs {
            let lhs_expr = &expressions[id];

            // Simplify `(a - b) + b` to `a`.
            if lhs_expr.op == Op::Subtract && op == Op::Add && lhs_expr.rhs == rhs {
                return lhs_expr.lhs;
            }
            // Simplify `(a + b) - b` to `a`.
            if lhs_expr.op == Op::Add && op == Op::Subtract && lhs_expr.rhs == rhs {
                return lhs_expr.lhs;
            }
            // Simplify `(a + b) - a` to `b`.
            if lhs_expr.op == Op::Add && op == Op::Subtract && lhs_expr.lhs == rhs {
                return lhs_expr.rhs;
            }
        }

        if let BcbCounter::Expression { id } = rhs {
            let rhs_expr = &expressions[id];

            // Simplify `a + (b - a)` to `b`.
            if op == Op::Add && rhs_expr.op == Op::Subtract && lhs == rhs_expr.rhs {
                return rhs_expr.lhs;
            }
            // Simplify `a - (a - b)` to `b`.
            if op == Op::Subtract && rhs_expr.op == Op::Subtract && lhs == rhs_expr.lhs {
                return rhs_expr.rhs;
            }
        }

        // Simplification failed, so actually create the new expression.
        let id = expressions.push(new_expr);
        BcbCounter::Expression { id }
    }

    /// Creates a counter that is the sum of the given counters.
    ///
    /// Returns `None` if the given list of counters was empty.
    fn make_sum(&mut self, counters: &[BcbCounter]) -> Option<BcbCounter> {
        counters
            .iter()
            .copied()
            .reduce(|accum, counter| self.make_expression(accum, Op::Add, counter))
    }

    pub(super) fn num_counters(&self) -> usize {
        self.counter_increment_sites.len()
    }

    fn set_bcb_counter(&mut self, bcb: BasicCoverageBlock, counter_kind: BcbCounter) -> BcbCounter {
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
        let old_len = self.expressions.len();
        let expressions = self
            .expressions
            .into_iter()
            .map(|BcbExpression { lhs, op, rhs }| Expression {
                lhs: lhs.as_term(),
                op,
                rhs: rhs.as_term(),
            })
            .collect::<IndexVec<ExpressionId, _>>();

        // Expression IDs are indexes into this vector, so make sure we didn't
        // accidentally invalidate them by changing its length.
        assert_eq!(old_len, expressions.len());
        expressions
    }
}

/// Helper struct that allows counter creation to inspect the BCB graph.
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

    fn make_bcb_counters(&mut self, bcb_needs_counter: impl Fn(BasicCoverageBlock) -> bool) {
        debug!("make_bcb_counters(): adding a counter or expression to each BasicCoverageBlock");

        // Traverse the coverage graph, ensuring that every node that needs a
        // coverage counter has one.
        //
        // The traversal tries to ensure that, when a loop is encountered, all
        // nodes within the loop are visited before visiting any nodes outside
        // the loop. It also keeps track of which loop(s) the traversal is
        // currently inside.
        let mut traversal = TraverseCoverageGraphWithLoops::new(self.basic_coverage_blocks);
        while let Some(bcb) = traversal.next() {
            let _span = debug_span!("traversal", ?bcb).entered();
            if bcb_needs_counter(bcb) {
                self.make_node_counter_and_out_edge_counters(&traversal, bcb);
            }
        }

        assert!(
            traversal.is_complete(),
            "`TraverseCoverageGraphWithLoops` missed some `BasicCoverageBlock`s: {:?}",
            traversal.unvisited(),
        );
    }

    /// Make sure the given node has a node counter, and then make sure each of
    /// its out-edges has an edge counter (if appropriate).
    #[instrument(level = "debug", skip(self, traversal))]
    fn make_node_counter_and_out_edge_counters(
        &mut self,
        traversal: &TraverseCoverageGraphWithLoops<'_>,
        from_bcb: BasicCoverageBlock,
    ) {
        // First, ensure that this node has a counter of some kind.
        // We might also use that counter to compute one of the out-edge counters.
        let node_counter = self.get_or_make_node_counter(from_bcb);

        let successors = self.basic_coverage_blocks.successors[from_bcb].as_slice();

        // If this node's out-edges won't sum to the node's counter,
        // then there's no reason to create edge counters here.
        if !self.basic_coverage_blocks[from_bcb].is_out_summable {
            return;
        }

        // Determine the set of out-edges that don't yet have edge counters.
        let candidate_successors = self.basic_coverage_blocks.successors[from_bcb]
            .iter()
            .copied()
            .filter(|&to_bcb| self.edge_has_no_counter(from_bcb, to_bcb))
            .collect::<Vec<_>>();
        debug!(?candidate_successors);

        // If there are out-edges without counters, choose one to be given an expression
        // (computed from this node and the other out-edges) instead of a physical counter.
        let Some(expression_to_bcb) =
            self.choose_out_edge_for_expression(traversal, &candidate_successors)
        else {
            return;
        };

        // For each out-edge other than the one that was chosen to get an expression,
        // ensure that it has a counter (existing counter/expression or a new counter),
        // and accumulate the corresponding counters into a single sum expression.
        let other_out_edge_counters = successors
            .iter()
            .copied()
            // Skip the chosen edge, since we'll calculate its count from this sum.
            .filter(|&to_bcb| to_bcb != expression_to_bcb)
            .map(|to_bcb| self.get_or_make_edge_counter(from_bcb, to_bcb))
            .collect::<Vec<_>>();
        let Some(sum_of_all_other_out_edges) =
            self.coverage_counters.make_sum(&other_out_edge_counters)
        else {
            return;
        };

        // Now create an expression for the chosen edge, by taking the counter
        // for its source node and subtracting the sum of its sibling out-edges.
        let expression = self.coverage_counters.make_expression(
            node_counter,
            Op::Subtract,
            sum_of_all_other_out_edges,
        );

        debug!("{expression_to_bcb:?} gets an expression: {expression:?}");
        if let Some(sole_pred) = self.basic_coverage_blocks.sole_predecessor(expression_to_bcb) {
            // This edge normally wouldn't get its own counter, so attach the expression
            // to its target node instead, so that `edge_has_no_counter` can see it.
            assert_eq!(sole_pred, from_bcb);
            self.coverage_counters.set_bcb_counter(expression_to_bcb, expression);
        } else {
            self.coverage_counters.set_bcb_edge_counter(from_bcb, expression_to_bcb, expression);
        }
    }

    #[instrument(level = "debug", skip(self))]
    fn get_or_make_node_counter(&mut self, bcb: BasicCoverageBlock) -> BcbCounter {
        // If the BCB already has a counter, return it.
        if let Some(counter_kind) = self.coverage_counters.bcb_counters[bcb] {
            debug!("{bcb:?} already has a counter: {counter_kind:?}");
            return counter_kind;
        }

        let predecessors = self.basic_coverage_blocks.predecessors[bcb].as_slice();

        // Handle cases where we can't compute a node's count from its in-edges:
        // - START_BCB has no in-edges, so taking the sum would panic (or be wrong).
        // - For nodes with one in-edge, or that directly loop to themselves,
        //   trying to get the in-edge counts would require this node's counter,
        //   leading to infinite recursion.
        if predecessors.len() <= 1 || predecessors.contains(&bcb) {
            debug!(?bcb, ?predecessors, "node has <=1 predecessors or is its own predecessor");
            return self.coverage_counters.make_phys_node_counter(bcb);
        }

        // A BCB with multiple incoming edges can compute its count by ensuring that counters
        // exist for each of those edges, and then adding them up to get a total count.
        let in_edge_counters = predecessors
            .iter()
            .copied()
            .map(|from_bcb| self.get_or_make_edge_counter(from_bcb, bcb))
            .collect::<Vec<_>>();
        let sum_of_in_edges: BcbCounter = self
            .coverage_counters
            .make_sum(&in_edge_counters)
            .expect("there must be at least one in-edge");

        debug!("{bcb:?} gets a new counter (sum of predecessor counters): {sum_of_in_edges:?}");
        self.coverage_counters.set_bcb_counter(bcb, sum_of_in_edges)
    }

    #[instrument(level = "debug", skip(self))]
    fn get_or_make_edge_counter(
        &mut self,
        from_bcb: BasicCoverageBlock,
        to_bcb: BasicCoverageBlock,
    ) -> BcbCounter {
        // If the target node has exactly one in-edge (i.e. this one), then just
        // use the node's counter, since it will have the same value.
        if let Some(sole_pred) = self.basic_coverage_blocks.sole_predecessor(to_bcb) {
            assert_eq!(sole_pred, from_bcb);
            // This call must take care not to invoke `get_or_make_edge` for
            // this edge, since that would result in infinite recursion!
            return self.get_or_make_node_counter(to_bcb);
        }

        // If the source node has exactly one out-edge (i.e. this one) and would have
        // the same execution count as that edge, then just use the node's counter.
        if let Some(simple_succ) = self.basic_coverage_blocks.simple_successor(from_bcb) {
            assert_eq!(simple_succ, to_bcb);
            return self.get_or_make_node_counter(from_bcb);
        }

        // If the edge already has a counter, return it.
        if let Some(&counter_kind) =
            self.coverage_counters.bcb_edge_counters.get(&(from_bcb, to_bcb))
        {
            debug!("Edge {from_bcb:?}->{to_bcb:?} already has a counter: {counter_kind:?}");
            return counter_kind;
        }

        // Make a new counter to count this edge.
        self.coverage_counters.make_phys_edge_counter(from_bcb, to_bcb)
    }

    /// Given a set of candidate out-edges (represented by their successor node),
    /// choose one to be given a counter expression instead of a physical counter.
    fn choose_out_edge_for_expression(
        &self,
        traversal: &TraverseCoverageGraphWithLoops<'_>,
        candidate_successors: &[BasicCoverageBlock],
    ) -> Option<BasicCoverageBlock> {
        // Try to find a candidate that leads back to the top of a loop,
        // because reloop edges tend to be executed more times than loop-exit edges.
        if let Some(reloop_target) = self.find_good_reloop_edge(traversal, &candidate_successors) {
            debug!("Selecting reloop target {reloop_target:?} to get an expression");
            return Some(reloop_target);
        }

        // We couldn't identify a "good" edge, so just choose an arbitrary one.
        let arbitrary_target = candidate_successors.first().copied()?;
        debug!(?arbitrary_target, "selecting arbitrary out-edge to get an expression");
        Some(arbitrary_target)
    }

    /// Given a set of candidate out-edges (represented by their successor node),
    /// tries to find one that leads back to the top of a loop.
    ///
    /// Reloop edges are good candidates for counter expressions, because they
    /// will tend to be executed more times than a loop-exit edge, so it's nice
    /// for them to be able to avoid a physical counter increment.
    fn find_good_reloop_edge(
        &self,
        traversal: &TraverseCoverageGraphWithLoops<'_>,
        candidate_successors: &[BasicCoverageBlock],
    ) -> Option<BasicCoverageBlock> {
        // If there are no candidates, avoid iterating over the loop stack.
        if candidate_successors.is_empty() {
            return None;
        }

        // Consider each loop on the current traversal context stack, top-down.
        for reloop_bcbs in traversal.reloop_bcbs_per_loop() {
            // Try to find a candidate edge that doesn't exit this loop.
            for &target_bcb in candidate_successors {
                // An edge is a reloop edge if its target dominates any BCB that has
                // an edge back to the loop header. (Otherwise it's an exit edge.)
                let is_reloop_edge = reloop_bcbs.iter().any(|&reloop_bcb| {
                    self.basic_coverage_blocks.dominates(target_bcb, reloop_bcb)
                });
                if is_reloop_edge {
                    // We found a good out-edge to be given an expression.
                    return Some(target_bcb);
                }
            }

            // All of the candidate edges exit this loop, so keep looking
            // for a good reloop edge for one of the outer loops.
        }

        None
    }

    #[inline]
    fn edge_has_no_counter(
        &self,
        from_bcb: BasicCoverageBlock,
        to_bcb: BasicCoverageBlock,
    ) -> bool {
        let edge_counter =
            if let Some(sole_pred) = self.basic_coverage_blocks.sole_predecessor(to_bcb) {
                assert_eq!(sole_pred, from_bcb);
                self.coverage_counters.bcb_counters[to_bcb]
            } else {
                self.coverage_counters.bcb_edge_counters.get(&(from_bcb, to_bcb)).copied()
            };

        edge_counter.is_none()
    }
}
