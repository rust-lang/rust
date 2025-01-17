use std::cmp::Ordering;
use std::fmt::{self, Debug};

use either::Either;
use itertools::Itertools;
use rustc_data_structures::captures::Captures;
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::graph::DirectedGraph;
use rustc_index::IndexVec;
use rustc_index::bit_set::DenseBitSet;
use rustc_middle::mir::coverage::{CounterId, CovTerm, Expression, ExpressionId, Op};

use crate::coverage::counters::balanced_flow::BalancedFlowGraph;
use crate::coverage::counters::iter_nodes::IterNodes;
use crate::coverage::counters::node_flow::{CounterTerm, MergedNodeFlowGraph, NodeCounters};
use crate::coverage::graph::{BasicCoverageBlock, CoverageGraph};

mod balanced_flow;
mod iter_nodes;
mod node_flow;
mod union_find;

/// The coverage counter or counter expression associated with a particular
/// BCB node or BCB edge.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum BcbCounter {
    Counter { id: CounterId },
    Expression { id: ExpressionId },
}

impl BcbCounter {
    fn as_term(&self) -> CovTerm {
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

/// Enum representing either a node or an edge in the coverage graph.
///
/// FIXME(#135481): This enum is no longer needed now that we only instrument
/// nodes and not edges. It can be removed in a subsequent PR.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) enum Site {
    Node { bcb: BasicCoverageBlock },
}

/// Generates and stores coverage counter and coverage expression information
/// associated with nodes/edges in the BCB graph.
pub(super) struct CoverageCounters {
    /// List of places where a counter-increment statement should be injected
    /// into MIR, each with its corresponding counter ID.
    counter_increment_sites: IndexVec<CounterId, Site>,

    /// Coverage counters/expressions that are associated with individual BCBs.
    node_counters: IndexVec<BasicCoverageBlock, Option<BcbCounter>>,

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
        graph: &CoverageGraph,
        bcb_needs_counter: &DenseBitSet<BasicCoverageBlock>,
    ) -> Self {
        let balanced_graph = BalancedFlowGraph::for_graph(graph, |n| !graph[n].is_out_summable);
        let merged_graph = MergedNodeFlowGraph::for_balanced_graph(&balanced_graph);

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
        let node_counters = merged_graph.make_node_counters(&nodes);

        Transcriber::new(graph.num_nodes(), node_counters).transcribe_counters(bcb_needs_counter)
    }

    fn with_num_bcbs(num_bcbs: usize) -> Self {
        Self {
            counter_increment_sites: IndexVec::new(),
            node_counters: IndexVec::from_elem_n(None, num_bcbs),
            expressions: IndexVec::new(),
            expressions_memo: FxHashMap::default(),
        }
    }

    /// Creates a new physical counter for a BCB node or edge.
    fn make_phys_counter(&mut self, site: Site) -> BcbCounter {
        let id = self.counter_increment_sites.push(site);
        BcbCounter::Counter { id }
    }

    fn make_expression(&mut self, lhs: BcbCounter, op: Op, rhs: BcbCounter) -> BcbCounter {
        let new_expr = BcbExpression { lhs, op, rhs };
        *self.expressions_memo.entry(new_expr).or_insert_with(|| {
            let id = self.expressions.push(new_expr);
            BcbCounter::Expression { id }
        })
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

    /// Creates a counter whose value is `lhs - SUM(rhs)`.
    fn make_subtracted_sum(&mut self, lhs: BcbCounter, rhs: &[BcbCounter]) -> BcbCounter {
        let Some(rhs_sum) = self.make_sum(rhs) else { return lhs };
        self.make_expression(lhs, Op::Subtract, rhs_sum)
    }

    pub(super) fn num_counters(&self) -> usize {
        self.counter_increment_sites.len()
    }

    fn set_node_counter(&mut self, bcb: BasicCoverageBlock, counter: BcbCounter) -> BcbCounter {
        let existing = self.node_counters[bcb].replace(counter);
        assert!(
            existing.is_none(),
            "node {bcb:?} already has a counter: {existing:?} => {counter:?}"
        );
        counter
    }

    pub(super) fn term_for_bcb(&self, bcb: BasicCoverageBlock) -> Option<CovTerm> {
        self.node_counters[bcb].map(|counter| counter.as_term())
    }

    /// Returns an iterator over all the nodes/edges in the coverage graph that
    /// should have a counter-increment statement injected into MIR, along with
    /// each site's corresponding counter ID.
    pub(super) fn counter_increment_sites(
        &self,
    ) -> impl Iterator<Item = (CounterId, Site)> + Captures<'_> {
        self.counter_increment_sites.iter_enumerated().map(|(id, &site)| (id, site))
    }

    /// Returns an iterator over the subset of BCB nodes that have been associated
    /// with a counter *expression*, along with the ID of that expression.
    pub(super) fn bcb_nodes_with_coverage_expressions(
        &self,
    ) -> impl Iterator<Item = (BasicCoverageBlock, ExpressionId)> + Captures<'_> {
        self.node_counters.iter_enumerated().filter_map(|(bcb, &counter)| match counter {
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

struct Transcriber {
    old: NodeCounters<BasicCoverageBlock>,
    new: CoverageCounters,
    phys_counter_for_site: FxHashMap<Site, BcbCounter>,
}

impl Transcriber {
    fn new(num_nodes: usize, old: NodeCounters<BasicCoverageBlock>) -> Self {
        Self {
            old,
            new: CoverageCounters::with_num_bcbs(num_nodes),
            phys_counter_for_site: FxHashMap::default(),
        }
    }

    fn transcribe_counters(
        mut self,
        bcb_needs_counter: &DenseBitSet<BasicCoverageBlock>,
    ) -> CoverageCounters {
        for bcb in bcb_needs_counter.iter() {
            let site = Site::Node { bcb };
            let (mut pos, mut neg): (Vec<_>, Vec<_>) =
                self.old.counter_expr(bcb).iter().partition_map(
                    |&CounterTerm { node, op }| match op {
                        Op::Add => Either::Left(node),
                        Op::Subtract => Either::Right(node),
                    },
                );

            if pos.is_empty() {
                // If we somehow end up with no positive terms, fall back to
                // creating a physical counter. There's no known way for this
                // to happen, but we can avoid an ICE if it does.
                debug_assert!(false, "{site:?} has no positive counter terms");
                pos = vec![bcb];
                neg = vec![];
            }

            pos.sort();
            neg.sort();

            let mut new_counters_for_sites = |sites: Vec<BasicCoverageBlock>| {
                sites
                    .into_iter()
                    .map(|node| self.ensure_phys_counter(Site::Node { bcb: node }))
                    .collect::<Vec<_>>()
            };
            let mut pos = new_counters_for_sites(pos);
            let mut neg = new_counters_for_sites(neg);

            pos.sort();
            neg.sort();

            let pos_counter = self.new.make_sum(&pos).expect("`pos` should not be empty");
            let new_counter = self.new.make_subtracted_sum(pos_counter, &neg);
            self.new.set_node_counter(bcb, new_counter);
        }

        self.new
    }

    fn ensure_phys_counter(&mut self, site: Site) -> BcbCounter {
        *self.phys_counter_for_site.entry(site).or_insert_with(|| self.new.make_phys_counter(site))
    }
}
