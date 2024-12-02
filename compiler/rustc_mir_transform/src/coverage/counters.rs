use std::cmp::Ordering;
use std::fmt::{self, Debug};

use rustc_data_structures::captures::Captures;
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::graph::DirectedGraph;
use rustc_index::IndexVec;
use rustc_index::bit_set::BitSet;
use rustc_middle::mir::coverage::{CounterId, CovTerm, Expression, ExpressionId, Op};
use tracing::{debug, debug_span, instrument};

use crate::coverage::graph::{BasicCoverageBlock, CoverageGraph, TraverseCoverageGraphWithLoops};

#[cfg(test)]
mod tests;

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
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) enum Site {
    Node { bcb: BasicCoverageBlock },
    Edge { from_bcb: BasicCoverageBlock, to_bcb: BasicCoverageBlock },
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
        bcb_needs_counter: &BitSet<BasicCoverageBlock>,
    ) -> Self {
        let mut builder = CountersBuilder::new(graph, bcb_needs_counter);
        builder.make_bcb_counters();

        builder.into_coverage_counters()
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

/// Symbolic representation of the coverage counter to be used for a particular
/// node or edge in the coverage graph. The same site counter can be used for
/// multiple sites, if they have been determined to have the same count.
#[derive(Clone, Copy, Debug)]
enum SiteCounter {
    /// A physical counter at some node/edge.
    Phys { site: Site },
    /// A counter expression for a node that takes the sum of all its in-edge
    /// counters.
    NodeSumExpr { bcb: BasicCoverageBlock },
    /// A counter expression for an edge that takes the counter of its source
    /// node, and subtracts the counters of all its sibling out-edges.
    EdgeDiffExpr { from_bcb: BasicCoverageBlock, to_bcb: BasicCoverageBlock },
}

/// Yields the graph successors of `from_bcb` that aren't `to_bcb`. This is
/// used when creating a counter expression for [`SiteCounter::EdgeDiffExpr`].
///
/// For example, in this diagram the sibling out-edge targets of edge `AC` are
/// the nodes `B` and `D`.
///
/// ```text
///    A
///  / | \
/// B  C  D
/// ```
fn sibling_out_edge_targets(
    graph: &CoverageGraph,
    from_bcb: BasicCoverageBlock,
    to_bcb: BasicCoverageBlock,
) -> impl Iterator<Item = BasicCoverageBlock> + Captures<'_> {
    graph.successors[from_bcb].iter().copied().filter(move |&t| t != to_bcb)
}

/// Helper struct that allows counter creation to inspect the BCB graph, and
/// the set of nodes that need counters.
struct CountersBuilder<'a> {
    graph: &'a CoverageGraph,
    bcb_needs_counter: &'a BitSet<BasicCoverageBlock>,

    site_counters: FxHashMap<Site, SiteCounter>,
}

impl<'a> CountersBuilder<'a> {
    fn new(graph: &'a CoverageGraph, bcb_needs_counter: &'a BitSet<BasicCoverageBlock>) -> Self {
        assert_eq!(graph.num_nodes(), bcb_needs_counter.domain_size());
        Self { graph, bcb_needs_counter, site_counters: FxHashMap::default() }
    }

    fn make_bcb_counters(&mut self) {
        debug!("make_bcb_counters(): adding a counter or expression to each BasicCoverageBlock");

        // Traverse the coverage graph, ensuring that every node that needs a
        // coverage counter has one.
        //
        // The traversal tries to ensure that, when a loop is encountered, all
        // nodes within the loop are visited before visiting any nodes outside
        // the loop.
        let mut traversal = TraverseCoverageGraphWithLoops::new(self.graph);
        while let Some(bcb) = traversal.next() {
            let _span = debug_span!("traversal", ?bcb).entered();
            if self.bcb_needs_counter.contains(bcb) {
                self.make_node_counter_and_out_edge_counters(bcb);
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
    #[instrument(level = "debug", skip(self))]
    fn make_node_counter_and_out_edge_counters(&mut self, from_bcb: BasicCoverageBlock) {
        // First, ensure that this node has a counter of some kind.
        // We might also use that counter to compute one of the out-edge counters.
        self.get_or_make_node_counter(from_bcb);

        // If this node's out-edges won't sum to the node's counter,
        // then there's no reason to create edge counters here.
        if !self.graph[from_bcb].is_out_summable {
            return;
        }

        // When choosing which out-edge should be given a counter expression, ignore edges that
        // already have counters, or could use the existing counter of their target node.
        let out_edge_has_counter = |to_bcb| {
            if self.site_counters.contains_key(&Site::Edge { from_bcb, to_bcb }) {
                return true;
            }
            self.graph.sole_predecessor(to_bcb) == Some(from_bcb)
                && self.site_counters.contains_key(&Site::Node { bcb: to_bcb })
        };

        // Determine the set of out-edges that could benefit from being given an expression.
        let candidate_successors = self.graph.successors[from_bcb]
            .iter()
            .copied()
            .filter(|&to_bcb| !out_edge_has_counter(to_bcb))
            .collect::<Vec<_>>();
        debug!(?candidate_successors);

        // If there are out-edges without counters, choose one to be given an expression
        // (computed from this node and the other out-edges) instead of a physical counter.
        let Some(to_bcb) = self.choose_out_edge_for_expression(from_bcb, &candidate_successors)
        else {
            return;
        };

        // For each out-edge other than the one that was chosen to get an expression,
        // ensure that it has a counter (existing counter/expression or a new counter).
        for target in sibling_out_edge_targets(self.graph, from_bcb, to_bcb) {
            self.get_or_make_edge_counter(from_bcb, target);
        }

        // Now create an expression for the chosen edge, by taking the counter
        // for its source node and subtracting the sum of its sibling out-edges.
        let counter = SiteCounter::EdgeDiffExpr { from_bcb, to_bcb };
        self.site_counters.insert(Site::Edge { from_bcb, to_bcb }, counter);
    }

    #[instrument(level = "debug", skip(self))]
    fn get_or_make_node_counter(&mut self, bcb: BasicCoverageBlock) -> SiteCounter {
        // If the BCB already has a counter, return it.
        if let Some(&counter) = self.site_counters.get(&Site::Node { bcb }) {
            debug!("{bcb:?} already has a counter: {counter:?}");
            return counter;
        }

        let counter = self.make_node_counter_inner(bcb);
        self.site_counters.insert(Site::Node { bcb }, counter);
        counter
    }

    fn make_node_counter_inner(&mut self, bcb: BasicCoverageBlock) -> SiteCounter {
        // If the node's sole in-edge already has a counter, use that.
        if let Some(sole_pred) = self.graph.sole_predecessor(bcb)
            && let Some(&edge_counter) =
                self.site_counters.get(&Site::Edge { from_bcb: sole_pred, to_bcb: bcb })
        {
            return edge_counter;
        }

        let predecessors = self.graph.predecessors[bcb].as_slice();

        // Handle cases where we can't compute a node's count from its in-edges:
        // - START_BCB has no in-edges, so taking the sum would panic (or be wrong).
        // - For nodes with one in-edge, or that directly loop to themselves,
        //   trying to get the in-edge counts would require this node's counter,
        //   leading to infinite recursion.
        if predecessors.len() <= 1 || predecessors.contains(&bcb) {
            debug!(?bcb, ?predecessors, "node has <=1 predecessors or is its own predecessor");
            let counter = SiteCounter::Phys { site: Site::Node { bcb } };
            debug!(?bcb, ?counter, "node gets a physical counter");
            return counter;
        }

        // A BCB with multiple incoming edges can compute its count by ensuring that counters
        // exist for each of those edges, and then adding them up to get a total count.
        for &from_bcb in predecessors {
            self.get_or_make_edge_counter(from_bcb, bcb);
        }
        let sum_of_in_edges = SiteCounter::NodeSumExpr { bcb };

        debug!("{bcb:?} gets a new counter (sum of predecessor counters): {sum_of_in_edges:?}");
        sum_of_in_edges
    }

    #[instrument(level = "debug", skip(self))]
    fn get_or_make_edge_counter(
        &mut self,
        from_bcb: BasicCoverageBlock,
        to_bcb: BasicCoverageBlock,
    ) -> SiteCounter {
        // If the edge already has a counter, return it.
        if let Some(&counter) = self.site_counters.get(&Site::Edge { from_bcb, to_bcb }) {
            debug!("Edge {from_bcb:?}->{to_bcb:?} already has a counter: {counter:?}");
            return counter;
        }

        let counter = self.make_edge_counter_inner(from_bcb, to_bcb);
        self.site_counters.insert(Site::Edge { from_bcb, to_bcb }, counter);
        counter
    }

    fn make_edge_counter_inner(
        &mut self,
        from_bcb: BasicCoverageBlock,
        to_bcb: BasicCoverageBlock,
    ) -> SiteCounter {
        // If the target node has exactly one in-edge (i.e. this one), then just
        // use the node's counter, since it will have the same value.
        if let Some(sole_pred) = self.graph.sole_predecessor(to_bcb) {
            assert_eq!(sole_pred, from_bcb);
            // This call must take care not to invoke `get_or_make_edge` for
            // this edge, since that would result in infinite recursion!
            return self.get_or_make_node_counter(to_bcb);
        }

        // If the source node has exactly one out-edge (i.e. this one) and would have
        // the same execution count as that edge, then just use the node's counter.
        if let Some(simple_succ) = self.graph.simple_successor(from_bcb) {
            assert_eq!(simple_succ, to_bcb);
            return self.get_or_make_node_counter(from_bcb);
        }

        // Make a new counter to count this edge.
        let counter = SiteCounter::Phys { site: Site::Edge { from_bcb, to_bcb } };
        debug!(?from_bcb, ?to_bcb, ?counter, "edge gets a physical counter");
        counter
    }

    /// Given a set of candidate out-edges (represented by their successor node),
    /// choose one to be given a counter expression instead of a physical counter.
    fn choose_out_edge_for_expression(
        &self,
        from_bcb: BasicCoverageBlock,
        candidate_successors: &[BasicCoverageBlock],
    ) -> Option<BasicCoverageBlock> {
        // Try to find a candidate that leads back to the top of a loop,
        // because reloop edges tend to be executed more times than loop-exit edges.
        if let Some(reloop_target) = self.find_good_reloop_edge(from_bcb, &candidate_successors) {
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
        from_bcb: BasicCoverageBlock,
        candidate_successors: &[BasicCoverageBlock],
    ) -> Option<BasicCoverageBlock> {
        // If there are no candidates, avoid iterating over the loop stack.
        if candidate_successors.is_empty() {
            return None;
        }

        // Consider each loop on the current traversal context stack, top-down.
        for loop_header_node in self.graph.loop_headers_containing(from_bcb) {
            // Try to find a candidate edge that doesn't exit this loop.
            for &target_bcb in candidate_successors {
                // An edge is a reloop edge if its target dominates any BCB that has
                // an edge back to the loop header. (Otherwise it's an exit edge.)
                let is_reloop_edge = self
                    .graph
                    .reloop_predecessors(loop_header_node)
                    .any(|reloop_bcb| self.graph.dominates(target_bcb, reloop_bcb));
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

    fn into_coverage_counters(self) -> CoverageCounters {
        Transcriber::new(&self).transcribe_counters()
    }
}

/// Helper struct for converting `CountersBuilder` into a final `CoverageCounters`.
struct Transcriber<'a> {
    old: &'a CountersBuilder<'a>,
    new: CoverageCounters,
    phys_counter_for_site: FxHashMap<Site, BcbCounter>,
}

impl<'a> Transcriber<'a> {
    fn new(old: &'a CountersBuilder<'a>) -> Self {
        Self {
            old,
            new: CoverageCounters::with_num_bcbs(old.graph.num_nodes()),
            phys_counter_for_site: FxHashMap::default(),
        }
    }

    fn transcribe_counters(mut self) -> CoverageCounters {
        for bcb in self.old.bcb_needs_counter.iter() {
            let site = Site::Node { bcb };
            let site_counter = self.site_counter(site);

            // Resolve the site counter into flat lists of nodes/edges whose
            // physical counts contribute to the counter for this node.
            // Distinguish between counts that will be added vs subtracted.
            let mut pos = vec![];
            let mut neg = vec![];
            self.push_resolved_sites(site_counter, &mut pos, &mut neg);

            // Simplify by cancelling out sites that appear on both sides.
            let (mut pos, mut neg) = sort_and_cancel(pos, neg);

            if pos.is_empty() {
                // If we somehow end up with no positive terms after cancellation,
                // fall back to creating a physical counter. There's no known way
                // for this to happen, but it's hard to confidently rule it out.
                debug_assert!(false, "{site:?} has no positive counter terms");
                pos = vec![Some(site)];
                neg = vec![];
            }

            let mut new_counters_for_sites = |sites: Vec<Option<Site>>| {
                sites
                    .into_iter()
                    .filter_map(|id| try { self.ensure_phys_counter(id?) })
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

    fn site_counter(&self, site: Site) -> SiteCounter {
        self.old.site_counters.get(&site).copied().unwrap_or_else(|| {
            // We should have already created all necessary site counters.
            // But if we somehow didn't, avoid crashing in release builds,
            // and just use an extra physical counter instead.
            debug_assert!(false, "{site:?} should have a counter");
            SiteCounter::Phys { site }
        })
    }

    fn ensure_phys_counter(&mut self, site: Site) -> BcbCounter {
        *self.phys_counter_for_site.entry(site).or_insert_with(|| self.new.make_phys_counter(site))
    }

    /// Resolves the given counter into flat lists of nodes/edges, whose counters
    /// will then be added and subtracted to form a counter expression.
    fn push_resolved_sites(&self, counter: SiteCounter, pos: &mut Vec<Site>, neg: &mut Vec<Site>) {
        match counter {
            SiteCounter::Phys { site } => pos.push(site),
            SiteCounter::NodeSumExpr { bcb } => {
                for &from_bcb in &self.old.graph.predecessors[bcb] {
                    let edge_counter = self.site_counter(Site::Edge { from_bcb, to_bcb: bcb });
                    self.push_resolved_sites(edge_counter, pos, neg);
                }
            }
            SiteCounter::EdgeDiffExpr { from_bcb, to_bcb } => {
                // First, add the count for `from_bcb`.
                let node_counter = self.site_counter(Site::Node { bcb: from_bcb });
                self.push_resolved_sites(node_counter, pos, neg);

                // Then subtract the counts for the other out-edges.
                for target in sibling_out_edge_targets(self.old.graph, from_bcb, to_bcb) {
                    let edge_counter = self.site_counter(Site::Edge { from_bcb, to_bcb: target });
                    // Swap `neg` and `pos` so that the counter is subtracted.
                    self.push_resolved_sites(edge_counter, neg, pos);
                }
            }
        }
    }
}

/// Given two lists:
/// - Sorts each list.
/// - Converts each list to `Vec<Option<T>>`.
/// - Scans for values that appear in both lists, and cancels them out by
///   replacing matching pairs of values with `None`.
fn sort_and_cancel<T: Ord>(mut pos: Vec<T>, mut neg: Vec<T>) -> (Vec<Option<T>>, Vec<Option<T>>) {
    pos.sort();
    neg.sort();

    // Convert to `Vec<Option<T>>`. If `T` has a niche, this should be zero-cost.
    let mut pos = pos.into_iter().map(Some).collect::<Vec<_>>();
    let mut neg = neg.into_iter().map(Some).collect::<Vec<_>>();

    // Scan through the lists using two cursors. When either cursor reaches the
    // end of its list, there can be no more equal pairs, so stop.
    let mut p = 0;
    let mut n = 0;
    while p < pos.len() && n < neg.len() {
        // If the values are equal, remove them and advance both cursors.
        // Otherwise, advance whichever cursor points to the lesser value.
        // (Choosing which cursor to advance relies on both lists being sorted.)
        match pos[p].cmp(&neg[n]) {
            Ordering::Less => p += 1,
            Ordering::Equal => {
                pos[p] = None;
                neg[n] = None;
                p += 1;
                n += 1;
            }
            Ordering::Greater => n += 1,
        }
    }

    (pos, neg)
}
