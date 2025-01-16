//! For each node in a control-flow graph, determines whether that node should
//! have a physical counter, or a counter expression that is derived from the
//! physical counters of other nodes.
//!
//! Based on the algorithm given in
//! "Optimal measurement points for program frequency counts"
//! (Knuth & Stevenson, 1973).

use rustc_data_structures::graph;
use rustc_index::bit_set::DenseBitSet;
use rustc_index::{Idx, IndexVec};
use rustc_middle::mir::coverage::Op;
use smallvec::SmallVec;

use crate::coverage::counters::iter_nodes::IterNodes;
use crate::coverage::counters::union_find::{FrozenUnionFind, UnionFind};

#[cfg(test)]
mod tests;

/// View of some underlying graph, in which each node's successors have been
/// merged into a single "supernode".
///
/// The resulting supernodes have no obvious meaning on their own.
/// However, merging successor nodes means that a node's out-edges can all
/// be combined into a single out-edge, whose flow is the same as the flow
/// (execution count) of its corresponding node in the original graph.
///
/// With all node flows now in the original graph now represented as edge flows
/// in the merged graph, it becomes possible to analyze the original node flows
/// using techniques for analyzing edge flows.
#[derive(Debug)]
pub(crate) struct MergedNodeFlowGraph<Node: Idx> {
    /// Maps each node to the supernode that contains it, indicated by some
    /// arbitrary "root" node that is part of that supernode.
    supernodes: FrozenUnionFind<Node>,
    /// For each node, stores the single supernode that all of its successors
    /// have been merged into.
    ///
    /// (Note that each node in a supernode can potentially have a _different_
    /// successor supernode from its peers.)
    succ_supernodes: IndexVec<Node, Node>,
}

impl<Node: Idx> MergedNodeFlowGraph<Node> {
    /// Creates a "merged" view of an underlying graph.
    ///
    /// The given graph is assumed to have [“balanced flow”](balanced-flow),
    /// though it does not necessarily have to be a `BalancedFlowGraph`.
    ///
    /// [balanced-flow]: `crate::coverage::counters::balanced_flow::BalancedFlowGraph`.
    pub(crate) fn for_balanced_graph<G>(graph: G) -> Self
    where
        G: graph::DirectedGraph<Node = Node> + graph::Successors,
    {
        let mut supernodes = UnionFind::<G::Node>::new(graph.num_nodes());

        // For each node, merge its successors into a single supernode, and
        // arbitrarily choose one of those successors to represent all of them.
        let successors = graph
            .iter_nodes()
            .map(|node| {
                graph
                    .successors(node)
                    .reduce(|a, b| supernodes.unify(a, b))
                    .expect("each node in a balanced graph must have at least one out-edge")
            })
            .collect::<IndexVec<G::Node, G::Node>>();

        // Now that unification is complete, freeze the supernode forest,
        // and resolve each arbitrarily-chosen successor to its canonical root.
        // (This avoids having to explicitly resolve them later.)
        let supernodes = supernodes.freeze();
        let succ_supernodes = successors.into_iter().map(|succ| supernodes.find(succ)).collect();

        Self { supernodes, succ_supernodes }
    }

    fn num_nodes(&self) -> usize {
        self.succ_supernodes.len()
    }

    fn is_supernode(&self, node: Node) -> bool {
        self.supernodes.find(node) == node
    }

    /// Using the information in this merged graph, together with a given
    /// permutation of all nodes in the graph, to create physical counters and
    /// counter expressions for each node in the underlying graph.
    ///
    /// The given list must contain exactly one copy of each node in the
    /// underlying balanced-flow graph. The order of nodes is used as a hint to
    /// influence counter allocation:
    /// - Earlier nodes are more likely to receive counter expressions.
    /// - Later nodes are more likely to receive physical counters.
    pub(crate) fn make_node_counters(&self, all_nodes_permutation: &[Node]) -> NodeCounters<Node> {
        let mut builder = SpantreeBuilder::new(self);

        for &node in all_nodes_permutation {
            builder.visit_node(node);
        }

        NodeCounters { counter_exprs: builder.finish() }
    }
}

/// End result of allocating physical counters and counter expressions for the
/// nodes of a graph.
#[derive(Debug)]
pub(crate) struct NodeCounters<Node: Idx> {
    counter_exprs: IndexVec<Node, CounterExprVec<Node>>,
}

impl<Node: Idx> NodeCounters<Node> {
    /// For the given node, returns the finished list of terms that represent
    /// its physical counter or counter expression. Always non-empty.
    ///
    /// If a node was given a physical counter, its "expression" will contain
    /// that counter as its sole element.
    pub(crate) fn counter_expr(&self, this: Node) -> &[CounterTerm<Node>] {
        self.counter_exprs[this].as_slice()
    }
}

#[derive(Debug)]
struct SpantreeEdge<Node> {
    /// If true, this edge in the spantree has been reversed an odd number of
    /// times, so all physical counters added to its node's counter expression
    /// need to be negated.
    is_reversed: bool,
    /// Each spantree edge is "claimed" by the (regular) node that caused it to
    /// be created. When a node with a physical counter traverses this edge,
    /// that counter is added to the claiming node's counter expression.
    claiming_node: Node,
    /// Supernode at the other end of this spantree edge. Transitively points
    /// to the "root" of this supernode's spantree component.
    span_parent: Node,
}

/// Part of a node's counter expression, which is a sum of counter terms.
#[derive(Debug)]
pub(crate) struct CounterTerm<Node> {
    /// Whether to add or subtract the value of the node's physical counter.
    pub(crate) op: Op,
    /// The node whose physical counter is represented by this term.
    pub(crate) node: Node,
}

/// Stores the list of counter terms that make up a node's counter expression.
type CounterExprVec<Node> = SmallVec<[CounterTerm<Node>; 2]>;

#[derive(Debug)]
struct SpantreeBuilder<'a, Node: Idx> {
    graph: &'a MergedNodeFlowGraph<Node>,
    is_unvisited: DenseBitSet<Node>,
    /// Links supernodes to each other, gradually forming a spanning tree of
    /// the merged-flow graph.
    ///
    /// A supernode without a span edge is the root of its component of the
    /// spantree. Nodes that aren't supernodes cannot have a spantree edge.
    span_edges: IndexVec<Node, Option<SpantreeEdge<Node>>>,
    /// Shared path buffer recycled by all calls to `yank_to_spantree_root`.
    yank_buffer: Vec<Node>,
    /// An in-progress counter expression for each node. Each expression is
    /// initially empty, and will be filled in as relevant nodes are visited.
    counter_exprs: IndexVec<Node, CounterExprVec<Node>>,
}

impl<'a, Node: Idx> SpantreeBuilder<'a, Node> {
    fn new(graph: &'a MergedNodeFlowGraph<Node>) -> Self {
        let num_nodes = graph.num_nodes();
        Self {
            graph,
            is_unvisited: DenseBitSet::new_filled(num_nodes),
            span_edges: IndexVec::from_fn_n(|_| None, num_nodes),
            yank_buffer: vec![],
            counter_exprs: IndexVec::from_fn_n(|_| SmallVec::new(), num_nodes),
        }
    }

    /// Given a supernode, finds the supernode that is the "root" of its
    /// spantree component. Two nodes that have the same spantree root are
    /// connected in the spantree.
    fn spantree_root(&self, this: Node) -> Node {
        debug_assert!(self.graph.is_supernode(this));

        match self.span_edges[this] {
            None => this,
            Some(SpantreeEdge { span_parent, .. }) => self.spantree_root(span_parent),
        }
    }

    /// Rotates edges in the spantree so that `this` is the root of its
    /// spantree component.
    fn yank_to_spantree_root(&mut self, this: Node) {
        debug_assert!(self.graph.is_supernode(this));

        // The rotation is done iteratively, by first traversing from `this` to
        // its root and storing the path in a buffer, and then traversing the
        // path buffer backwards to reverse all the edges.

        // Recycle the same path buffer for all calls to this method.
        let path_buf = &mut self.yank_buffer;
        path_buf.clear();
        path_buf.push(this);

        // Traverse the spantree until we reach a supernode that has no
        // span-parent, which must be the root.
        let mut curr = this;
        while let &Some(SpantreeEdge { span_parent, .. }) = &self.span_edges[curr] {
            path_buf.push(span_parent);
            curr = span_parent;
        }

        // For each spantree edge `a -> b` in the path that was just traversed,
        // reverse it to become `a <- b`, while preserving `claiming_node`.
        for &[a, b] in path_buf.array_windows::<2>().rev() {
            let SpantreeEdge { is_reversed, claiming_node, span_parent } = self.span_edges[a]
                .take()
                .expect("all nodes in the path (except the last) have a `span_parent`");
            debug_assert_eq!(span_parent, b);
            debug_assert!(self.span_edges[b].is_none());
            self.span_edges[b] =
                Some(SpantreeEdge { is_reversed: !is_reversed, claiming_node, span_parent: a });
        }

        // The result of the rotation is that `this` is now a spantree root.
        debug_assert!(self.span_edges[this].is_none());
    }

    /// Must be called exactly once for each node in the balanced-flow graph.
    fn visit_node(&mut self, this: Node) {
        // Assert that this node was unvisited, and mark it visited.
        assert!(self.is_unvisited.remove(this), "node has already been visited: {this:?}");

        // Get the supernode containing `this`, and make it the root of its
        // component of the spantree.
        let this_supernode = self.graph.supernodes.find(this);
        self.yank_to_spantree_root(this_supernode);

        // Get the supernode containing all of this's successors.
        let succ_supernode = self.graph.succ_supernodes[this];
        debug_assert!(self.graph.is_supernode(succ_supernode));

        // If two supernodes are already connected in the spantree, they will
        // have the same spantree root. (Each supernode is connected to itself.)
        if this_supernode != self.spantree_root(succ_supernode) {
            // Adding this node's flow edge to the spantree would cause two
            // previously-disconnected supernodes to become connected, so add
            // it. That spantree-edge is now "claimed" by this node.
            //
            // Claiming a spantree-edge means that this node will get a counter
            // expression instead of a physical counter. That expression is
            // currently empty, but will be built incrementally as the other
            // nodes are visited.
            self.span_edges[this_supernode] = Some(SpantreeEdge {
                is_reversed: false,
                claiming_node: this,
                span_parent: succ_supernode,
            });
        } else {
            // This node's flow edge would join two supernodes that are already
            // connected in the spantree (or are the same supernode). That would
            // create a cycle in the spantree, so don't add an edge.
            //
            // Instead, create a physical counter for this node, and add that
            // counter to all expressions on the path from `succ_supernode` to
            // `this_supernode`.

            // Instead of setting `this.measure = true` as in the original paper,
            // we just add the node's ID to its own "expression".
            self.counter_exprs[this].push(CounterTerm { node: this, op: Op::Add });

            // Walk the spantree from `this.successor` back to `this`. For each
            // spantree edge along the way, add this node's physical counter to
            // the counter expression of the node that claimed the spantree edge.
            let mut curr = succ_supernode;
            while curr != this_supernode {
                let &SpantreeEdge { is_reversed, claiming_node, span_parent } =
                    self.span_edges[curr].as_ref().unwrap();
                let op = if is_reversed { Op::Subtract } else { Op::Add };
                self.counter_exprs[claiming_node].push(CounterTerm { node: this, op });

                curr = span_parent;
            }
        }
    }

    /// Asserts that all nodes have been visited, and returns the computed
    /// counter expressions (made up of physical counters) for each node.
    fn finish(self) -> IndexVec<Node, CounterExprVec<Node>> {
        let Self { graph, is_unvisited, span_edges, yank_buffer: _, counter_exprs } = self;
        assert!(is_unvisited.is_empty(), "some nodes were never visited: {is_unvisited:?}");
        debug_assert!(
            span_edges
                .iter_enumerated()
                .all(|(node, span_edge)| { span_edge.is_some() <= graph.is_supernode(node) }),
            "only supernodes can have a span edge",
        );
        debug_assert!(
            counter_exprs.iter().all(|expr| !expr.is_empty()),
            "after visiting all nodes, every node should have a non-empty expression",
        );
        counter_exprs
    }
}
