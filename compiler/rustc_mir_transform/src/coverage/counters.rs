use std::cmp::Ordering;

use either::Either;
use itertools::Itertools;
use rustc_data_structures::captures::Captures;
use rustc_data_structures::fx::{FxHashMap, FxIndexMap};
use rustc_data_structures::graph::DirectedGraph;
use rustc_index::IndexVec;
use rustc_index::bit_set::DenseBitSet;
use rustc_middle::mir::coverage::{CounterId, CovTerm, Expression, ExpressionId, Op};

use crate::coverage::counters::balanced_flow::BalancedFlowGraph;
use crate::coverage::counters::node_flow::{
    CounterTerm, NodeCounters, make_node_counters, node_flow_data_for_balanced_graph,
};
use crate::coverage::graph::{BasicCoverageBlock, CoverageGraph};

mod balanced_flow;
mod node_flow;
mod union_find;

/// Ensures that each BCB node needing a counter has one, by creating physical
/// counters or counter expressions for nodes as required.
pub(super) fn make_bcb_counters(
    graph: &CoverageGraph,
    bcb_needs_counter: &DenseBitSet<BasicCoverageBlock>,
) -> CoverageCounters {
    // Create the derived graphs that are necessary for subsequent steps.
    let balanced_graph = BalancedFlowGraph::for_graph(graph, |n| !graph[n].is_out_summable);
    let node_flow_data = node_flow_data_for_balanced_graph(&balanced_graph);

    // Use those graphs to determine which nodes get physical counters, and how
    // to compute the execution counts of other nodes from those counters.
    let priority_list = make_node_flow_priority_list(graph, balanced_graph);
    let node_counters = make_node_counters(&node_flow_data, &priority_list);

    // Convert the counters into a form suitable for embedding into MIR.
    transcribe_counters(&node_counters, bcb_needs_counter)
}

/// Arranges the nodes in `balanced_graph` into a list, such that earlier nodes
/// take priority in being given a counter expression instead of a physical counter.
fn make_node_flow_priority_list(
    graph: &CoverageGraph,
    balanced_graph: BalancedFlowGraph<&CoverageGraph>,
) -> Vec<BasicCoverageBlock> {
    // A "reloop" node has exactly one out-edge, which jumps back to the top
    // of an enclosing loop. Reloop nodes are typically visited more times
    // than loop-exit nodes, so try to avoid giving them physical counters.
    let is_reloop_node = IndexVec::from_fn_n(
        |node| match graph.successors[node].as_slice() {
            &[succ] => graph.dominates(succ, node),
            _ => false,
        },
        graph.num_nodes(),
    );

    let mut nodes = balanced_graph.iter_nodes().rev().collect::<Vec<_>>();
    // The first node is the sink, which must not get a physical counter.
    assert_eq!(nodes[0], balanced_graph.sink);
    // Sort the real nodes, such that earlier (lesser) nodes take priority
    // in being given a counter expression instead of a physical counter.
    nodes[1..].sort_by(|&a, &b| {
        // Start with a dummy `Equal` to make the actual tests line up nicely.
        Ordering::Equal
            // Prefer a physical counter for return/yield nodes.
            .then_with(|| Ord::cmp(&graph[a].is_out_summable, &graph[b].is_out_summable))
            // Prefer an expression for reloop nodes (see definition above).
            .then_with(|| Ord::cmp(&is_reloop_node[a], &is_reloop_node[b]).reverse())
            // Otherwise, prefer a physical counter for dominating nodes.
            .then_with(|| graph.cmp_in_dominator_order(a, b).reverse())
    });
    nodes
}

// Converts node counters into a form suitable for embedding into MIR.
fn transcribe_counters(
    old: &NodeCounters<BasicCoverageBlock>,
    bcb_needs_counter: &DenseBitSet<BasicCoverageBlock>,
) -> CoverageCounters {
    let mut new = CoverageCounters::with_num_bcbs(bcb_needs_counter.domain_size());

    for bcb in bcb_needs_counter.iter() {
        // Our counter-creation algorithm doesn't guarantee that a node's list
        // of terms starts or ends with a positive term, so partition the
        // counters into "positive" and "negative" lists for easier handling.
        let (mut pos, mut neg): (Vec<_>, Vec<_>) =
            old.counter_terms[bcb].iter().partition_map(|&CounterTerm { node, op }| match op {
                Op::Add => Either::Left(node),
                Op::Subtract => Either::Right(node),
            });

        if pos.is_empty() {
            // If we somehow end up with no positive terms, fall back to
            // creating a physical counter. There's no known way for this
            // to happen, but we can avoid an ICE if it does.
            debug_assert!(false, "{bcb:?} has no positive counter terms");
            pos = vec![bcb];
            neg = vec![];
        }

        // These intermediate sorts are not strictly necessary, but were helpful
        // in reducing churn when switching to the current counter-creation scheme.
        // They also help to slightly decrease the overall size of the expression
        // table, due to more subexpressions being shared.
        pos.sort();
        neg.sort();

        let mut new_counters_for_sites = |sites: Vec<BasicCoverageBlock>| {
            sites.into_iter().map(|node| new.ensure_phys_counter(node)).collect::<Vec<_>>()
        };
        let mut pos = new_counters_for_sites(pos);
        let mut neg = new_counters_for_sites(neg);

        // These sorts are also not strictly necessary; see above.
        pos.sort();
        neg.sort();

        let pos_counter = new.make_sum(&pos).expect("`pos` should not be empty");
        let new_counter = new.make_subtracted_sum(pos_counter, &neg);
        new.set_node_counter(bcb, new_counter);
    }

    new
}

/// Generates and stores coverage counter and coverage expression information
/// associated with nodes in the coverage graph.
pub(super) struct CoverageCounters {
    /// List of places where a counter-increment statement should be injected
    /// into MIR, each with its corresponding counter ID.
    phys_counter_for_node: FxIndexMap<BasicCoverageBlock, CounterId>,
    next_counter_id: CounterId,

    /// Coverage counters/expressions that are associated with individual BCBs.
    node_counters: IndexVec<BasicCoverageBlock, Option<CovTerm>>,

    /// Table of expression data, associating each expression ID with its
    /// corresponding operator (+ or -) and its LHS/RHS operands.
    expressions: IndexVec<ExpressionId, Expression>,
    /// Remember expressions that have already been created (or simplified),
    /// so that we don't create unnecessary duplicates.
    expressions_memo: FxHashMap<Expression, CovTerm>,
}

impl CoverageCounters {
    fn with_num_bcbs(num_bcbs: usize) -> Self {
        Self {
            phys_counter_for_node: FxIndexMap::default(),
            next_counter_id: CounterId::ZERO,
            node_counters: IndexVec::from_elem_n(None, num_bcbs),
            expressions: IndexVec::new(),
            expressions_memo: FxHashMap::default(),
        }
    }

    /// Returns the physical counter for the given node, creating it if necessary.
    fn ensure_phys_counter(&mut self, bcb: BasicCoverageBlock) -> CovTerm {
        let id = *self.phys_counter_for_node.entry(bcb).or_insert_with(|| {
            let id = self.next_counter_id;
            self.next_counter_id = id + 1;
            id
        });
        CovTerm::Counter(id)
    }

    fn make_expression(&mut self, lhs: CovTerm, op: Op, rhs: CovTerm) -> CovTerm {
        let new_expr = Expression { lhs, op, rhs };
        *self.expressions_memo.entry(new_expr.clone()).or_insert_with(|| {
            let id = self.expressions.push(new_expr);
            CovTerm::Expression(id)
        })
    }

    /// Creates a counter that is the sum of the given counters.
    ///
    /// Returns `None` if the given list of counters was empty.
    fn make_sum(&mut self, counters: &[CovTerm]) -> Option<CovTerm> {
        counters
            .iter()
            .copied()
            .reduce(|accum, counter| self.make_expression(accum, Op::Add, counter))
    }

    /// Creates a counter whose value is `lhs - SUM(rhs)`.
    fn make_subtracted_sum(&mut self, lhs: CovTerm, rhs: &[CovTerm]) -> CovTerm {
        let Some(rhs_sum) = self.make_sum(rhs) else { return lhs };
        self.make_expression(lhs, Op::Subtract, rhs_sum)
    }

    pub(super) fn num_counters(&self) -> usize {
        let num_counters = self.phys_counter_for_node.len();
        assert_eq!(num_counters, self.next_counter_id.as_usize());
        num_counters
    }

    fn set_node_counter(&mut self, bcb: BasicCoverageBlock, counter: CovTerm) -> CovTerm {
        let existing = self.node_counters[bcb].replace(counter);
        assert!(
            existing.is_none(),
            "node {bcb:?} already has a counter: {existing:?} => {counter:?}"
        );
        counter
    }

    pub(super) fn term_for_bcb(&self, bcb: BasicCoverageBlock) -> Option<CovTerm> {
        self.node_counters[bcb]
    }

    /// Returns an iterator over all the nodes in the coverage graph that
    /// should have a counter-increment statement injected into MIR, along with
    /// each site's corresponding counter ID.
    pub(super) fn counter_increment_sites(
        &self,
    ) -> impl Iterator<Item = (CounterId, BasicCoverageBlock)> + Captures<'_> {
        self.phys_counter_for_node.iter().map(|(&site, &id)| (id, site))
    }

    /// Returns an iterator over the subset of BCB nodes that have been associated
    /// with a counter *expression*, along with the ID of that expression.
    pub(super) fn bcb_nodes_with_coverage_expressions(
        &self,
    ) -> impl Iterator<Item = (BasicCoverageBlock, ExpressionId)> + Captures<'_> {
        self.node_counters.iter_enumerated().filter_map(|(bcb, &counter)| match counter {
            // Yield the BCB along with its associated expression ID.
            Some(CovTerm::Expression(id)) => Some((bcb, id)),
            // This BCB is associated with a counter or nothing, so skip it.
            Some(CovTerm::Counter { .. } | CovTerm::Zero) | None => None,
        })
    }

    pub(super) fn into_expressions(self) -> IndexVec<ExpressionId, Expression> {
        self.expressions
    }
}
