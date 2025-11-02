use std::cmp::Ordering;

use either::Either;
use itertools::Itertools;
use rustc_data_structures::fx::{FxHashMap, FxIndexMap};
use rustc_data_structures::graph::DirectedGraph;
use rustc_index::IndexVec;
use rustc_index::bit_set::DenseBitSet;
use rustc_middle::mir::coverage::{CounterId, CovTerm, Expression, ExpressionId, Op};

use crate::coverage::counters::balanced_flow::BalancedFlowGraph;
use crate::coverage::counters::node_flow::{
    CounterTerm, NodeCounters, NodeFlowData, node_flow_data_for_balanced_graph,
};
use crate::coverage::graph::{BasicCoverageBlock, CoverageGraph};

mod balanced_flow;
pub(crate) mod node_flow;

/// Struct containing the results of [`prepare_bcb_counters_data`].
pub(crate) struct BcbCountersData {
    pub(crate) node_flow_data: NodeFlowData<BasicCoverageBlock>,
    pub(crate) priority_list: Vec<BasicCoverageBlock>,
}

/// Analyzes the coverage graph to create intermediate data structures that
/// will later be used (during codegen) to create physical counters or counter
/// expressions for each BCB node that needs one.
pub(crate) fn prepare_bcb_counters_data(graph: &CoverageGraph) -> BcbCountersData {
    // Create the derived graphs that are necessary for subsequent steps.
    let balanced_graph = BalancedFlowGraph::for_graph(graph, |n| !graph[n].is_out_summable);
    let node_flow_data = node_flow_data_for_balanced_graph(&balanced_graph);

    // Also create a "priority list" of coverage graph nodes, to help determine
    // which ones get physical counters or counter expressions. This needs to
    // be done now, because the later parts of the counter-creation process
    // won't have access to the original coverage graph.
    let priority_list = make_node_flow_priority_list(graph, balanced_graph);

    BcbCountersData { node_flow_data, priority_list }
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
    let is_reloop_node = IndexVec::<BasicCoverageBlock, _>::from_fn_n(
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
pub(crate) fn transcribe_counters(
    old: &NodeCounters<BasicCoverageBlock>,
    bcb_needs_counter: &DenseBitSet<BasicCoverageBlock>,
    bcbs_seen: &DenseBitSet<BasicCoverageBlock>,
) -> CoverageCounters {
    let mut new = CoverageCounters::with_num_bcbs(bcb_needs_counter.domain_size());

    for bcb in bcb_needs_counter.iter() {
        if !bcbs_seen.contains(bcb) {
            // This BCB's code was removed by MIR opts, so its counter is always zero.
            new.set_node_counter(bcb, CovTerm::Zero);
            continue;
        }

        // Our counter-creation algorithm doesn't guarantee that a node's list
        // of terms starts or ends with a positive term, so partition the
        // counters into "positive" and "negative" lists for easier handling.
        let (mut pos, mut neg): (Vec<_>, Vec<_>) = old.counter_terms[bcb]
            .iter()
            // Filter out any BCBs that were removed by MIR opts;
            // this treats them as having an execution count of 0.
            .filter(|term| bcbs_seen.contains(term.node))
            .partition_map(|&CounterTerm { node, op }| match op {
                Op::Add => Either::Left(node),
                Op::Subtract => Either::Right(node),
            });

        // These intermediate sorts are not strictly necessary, but were helpful
        // in reducing churn when switching to the current counter-creation scheme.
        // They also help to slightly decrease the overall size of the expression
        // table, due to more subexpressions being shared.
        pos.sort();
        neg.sort();

        let mut new_counters_for_sites = |sites: Vec<BasicCoverageBlock>| {
            sites.into_iter().map(|node| new.ensure_phys_counter(node)).collect::<Vec<_>>()
        };
        let pos = new_counters_for_sites(pos);
        let neg = new_counters_for_sites(neg);

        let pos_counter = new.make_sum(&pos).unwrap_or(CovTerm::Zero);
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
    pub(crate) phys_counter_for_node: FxIndexMap<BasicCoverageBlock, CounterId>,
    pub(crate) next_counter_id: CounterId,

    /// Coverage counters/expressions that are associated with individual BCBs.
    pub(crate) node_counters: IndexVec<BasicCoverageBlock, Option<CovTerm>>,

    /// Table of expression data, associating each expression ID with its
    /// corresponding operator (+ or -) and its LHS/RHS operands.
    pub(crate) expressions: IndexVec<ExpressionId, Expression>,
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

    fn set_node_counter(&mut self, bcb: BasicCoverageBlock, counter: CovTerm) -> CovTerm {
        let existing = self.node_counters[bcb].replace(counter);
        assert!(
            existing.is_none(),
            "node {bcb:?} already has a counter: {existing:?} => {counter:?}"
        );
        counter
    }
}
