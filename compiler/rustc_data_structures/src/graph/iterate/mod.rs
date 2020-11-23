use super::{DirectedGraph, WithNumNodes, WithStartNode, WithSuccessors};
use rustc_index::bit_set::BitSet;
use rustc_index::vec::IndexVec;
use std::ops::ControlFlow;

#[cfg(test)]
mod tests;

pub fn post_order_from<G: DirectedGraph + WithSuccessors + WithNumNodes>(
    graph: &G,
    start_node: G::Node,
) -> Vec<G::Node> {
    post_order_from_to(graph, start_node, None)
}

pub fn post_order_from_to<G: DirectedGraph + WithSuccessors + WithNumNodes>(
    graph: &G,
    start_node: G::Node,
    end_node: Option<G::Node>,
) -> Vec<G::Node> {
    let mut visited: IndexVec<G::Node, bool> = IndexVec::from_elem_n(false, graph.num_nodes());
    let mut result: Vec<G::Node> = Vec::with_capacity(graph.num_nodes());
    if let Some(end_node) = end_node {
        visited[end_node] = true;
    }
    post_order_walk(graph, start_node, &mut result, &mut visited);
    result
}

fn post_order_walk<G: DirectedGraph + WithSuccessors + WithNumNodes>(
    graph: &G,
    node: G::Node,
    result: &mut Vec<G::Node>,
    visited: &mut IndexVec<G::Node, bool>,
) {
    struct PostOrderFrame<Node, Iter> {
        node: Node,
        iter: Iter,
    }

    if visited[node] {
        return;
    }

    let mut stack = vec![PostOrderFrame { node, iter: graph.successors(node) }];

    'recurse: while let Some(frame) = stack.last_mut() {
        let node = frame.node;
        visited[node] = true;

        while let Some(successor) = frame.iter.next() {
            if !visited[successor] {
                stack.push(PostOrderFrame { node: successor, iter: graph.successors(successor) });
                continue 'recurse;
            }
        }

        let _ = stack.pop();
        result.push(node);
    }
}

pub fn reverse_post_order<G: DirectedGraph + WithSuccessors + WithNumNodes>(
    graph: &G,
    start_node: G::Node,
) -> Vec<G::Node> {
    let mut vec = post_order_from(graph, start_node);
    vec.reverse();
    vec
}

/// A "depth-first search" iterator for a directed graph.
pub struct DepthFirstSearch<'graph, G>
where
    G: ?Sized + DirectedGraph + WithNumNodes + WithSuccessors,
{
    graph: &'graph G,
    stack: Vec<G::Node>,
    visited: BitSet<G::Node>,
}

impl<G> DepthFirstSearch<'graph, G>
where
    G: ?Sized + DirectedGraph + WithNumNodes + WithSuccessors,
{
    pub fn new(graph: &'graph G) -> Self {
        Self { graph, stack: vec![], visited: BitSet::new_empty(graph.num_nodes()) }
    }

    /// Version of `push_start_node` that is convenient for chained
    /// use.
    pub fn with_start_node(mut self, start_node: G::Node) -> Self {
        self.push_start_node(start_node);
        self
    }

    /// Pushes another start node onto the stack. If the node
    /// has not already been visited, then you will be able to
    /// walk its successors (and so forth) after the current
    /// contents of the stack are drained. If multiple start nodes
    /// are added into the walk, then their mutual successors
    /// will all be walked. You can use this method once the
    /// iterator has been completely drained to add additional
    /// start nodes.
    pub fn push_start_node(&mut self, start_node: G::Node) {
        if self.visited.insert(start_node) {
            self.stack.push(start_node);
        }
    }

    /// Searches all nodes reachable from the current start nodes.
    /// This is equivalent to just invoke `next` repeatedly until
    /// you get a `None` result.
    pub fn complete_search(&mut self) {
        while let Some(_) = self.next() {}
    }

    /// Returns true if node has been visited thus far.
    /// A node is considered "visited" once it is pushed
    /// onto the internal stack; it may not yet have been yielded
    /// from the iterator. This method is best used after
    /// the iterator is completely drained.
    pub fn visited(&self, node: G::Node) -> bool {
        self.visited.contains(node)
    }
}

impl<G> std::fmt::Debug for DepthFirstSearch<'_, G>
where
    G: ?Sized + DirectedGraph + WithNumNodes + WithSuccessors,
{
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut f = fmt.debug_set();
        for n in self.visited.iter() {
            f.entry(&n);
        }
        f.finish()
    }
}

impl<G> Iterator for DepthFirstSearch<'_, G>
where
    G: ?Sized + DirectedGraph + WithNumNodes + WithSuccessors,
{
    type Item = G::Node;

    fn next(&mut self) -> Option<G::Node> {
        let DepthFirstSearch { stack, visited, graph } = self;
        let n = stack.pop()?;
        stack.extend(graph.successors(n).filter(|&m| visited.insert(m)));
        Some(n)
    }
}

/// The status of a node in the depth-first search.
///
/// See the documentation of `TriColorDepthFirstSearch` to see how a node's status is updated
/// during DFS.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeStatus {
    /// This node has been examined by the depth-first search but is not yet `Settled`.
    ///
    /// Also referred to as "gray" or "discovered" nodes in [CLR].
    ///
    /// [CLR]: https://en.wikipedia.org/wiki/Introduction_to_Algorithms
    Visited,

    /// This node and all nodes reachable from it have been examined by the depth-first search.
    ///
    /// Also referred to as "black" or "finished" nodes in [CLR].
    ///
    /// [CLR]: https://en.wikipedia.org/wiki/Introduction_to_Algorithms
    Settled,
}

struct Event<N> {
    node: N,
    becomes: NodeStatus,
}

/// A depth-first search that also tracks when all successors of a node have been examined.
///
/// This is based on the DFS described in [Introduction to Algorithms (1st ed.)][CLR], hereby
/// referred to as **CLR**. However, we use the terminology in [`NodeStatus`] above instead of
/// "discovered"/"finished" or "white"/"grey"/"black". Each node begins the search with no status,
/// becomes `Visited` when it is first examined by the DFS and is `Settled` when all nodes
/// reachable from it have been examined. This allows us to differentiate between "tree", "back"
/// and "forward" edges (see [`TriColorVisitor::node_examined`]).
///
/// Unlike the pseudocode in [CLR], this implementation is iterative and does not use timestamps.
/// We accomplish this by storing `Event`s on the stack that result in a (possible) state change
/// for each node. A `Visited` event signifies that we should examine this node if it has not yet
/// been `Visited` or `Settled`. When a node is examined for the first time, we mark it as
/// `Visited` and push a `Settled` event for it on stack followed by `Visited` events for all of
/// its predecessors, scheduling them for examination. Multiple `Visited` events for a single node
/// may exist on the stack simultaneously if a node has multiple predecessors, but only one
/// `Settled` event will ever be created for each node. After all `Visited` events for a node's
/// successors have been popped off the stack (as well as any new events triggered by visiting
/// those successors), we will pop off that node's `Settled` event.
///
/// [CLR]: https://en.wikipedia.org/wiki/Introduction_to_Algorithms
pub struct TriColorDepthFirstSearch<'graph, G>
where
    G: ?Sized + DirectedGraph + WithNumNodes + WithSuccessors,
{
    graph: &'graph G,
    stack: Vec<Event<G::Node>>,
    visited: BitSet<G::Node>,
    settled: BitSet<G::Node>,
}

impl<G> TriColorDepthFirstSearch<'graph, G>
where
    G: ?Sized + DirectedGraph + WithNumNodes + WithSuccessors,
{
    pub fn new(graph: &'graph G) -> Self {
        TriColorDepthFirstSearch {
            graph,
            stack: vec![],
            visited: BitSet::new_empty(graph.num_nodes()),
            settled: BitSet::new_empty(graph.num_nodes()),
        }
    }

    /// Performs a depth-first search, starting from the given `root`.
    ///
    /// This won't visit nodes that are not reachable from `root`.
    pub fn run_from<V>(mut self, root: G::Node, visitor: &mut V) -> Option<V::BreakVal>
    where
        V: TriColorVisitor<G>,
    {
        use NodeStatus::{Settled, Visited};

        self.stack.push(Event { node: root, becomes: Visited });

        loop {
            match self.stack.pop()? {
                Event { node, becomes: Settled } => {
                    let not_previously_settled = self.settled.insert(node);
                    assert!(not_previously_settled, "A node should be settled exactly once");
                    if let ControlFlow::Break(val) = visitor.node_settled(node) {
                        return Some(val);
                    }
                }

                Event { node, becomes: Visited } => {
                    let not_previously_visited = self.visited.insert(node);
                    let prior_status = if not_previously_visited {
                        None
                    } else if self.settled.contains(node) {
                        Some(Settled)
                    } else {
                        Some(Visited)
                    };

                    if let ControlFlow::Break(val) = visitor.node_examined(node, prior_status) {
                        return Some(val);
                    }

                    // If this node has already been examined, we are done.
                    if prior_status.is_some() {
                        continue;
                    }

                    // Otherwise, push a `Settled` event for this node onto the stack, then
                    // schedule its successors for examination.
                    self.stack.push(Event { node, becomes: Settled });
                    for succ in self.graph.successors(node) {
                        if !visitor.ignore_edge(node, succ) {
                            self.stack.push(Event { node: succ, becomes: Visited });
                        }
                    }
                }
            }
        }
    }
}

impl<G> TriColorDepthFirstSearch<'graph, G>
where
    G: ?Sized + DirectedGraph + WithNumNodes + WithSuccessors + WithStartNode,
{
    /// Performs a depth-first search, starting from `G::start_node()`.
    ///
    /// This won't visit nodes that are not reachable from the start node.
    pub fn run_from_start<V>(self, visitor: &mut V) -> Option<V::BreakVal>
    where
        V: TriColorVisitor<G>,
    {
        let root = self.graph.start_node();
        self.run_from(root, visitor)
    }
}

/// What to do when a node is examined or becomes `Settled` during DFS.
pub trait TriColorVisitor<G>
where
    G: ?Sized + DirectedGraph,
{
    /// The value returned by this search.
    type BreakVal;

    /// Called when a node is examined by the depth-first search.
    ///
    /// By checking the value of `prior_status`, this visitor can determine whether the edge
    /// leading to this node was a tree edge (`None`), forward edge (`Some(Settled)`) or back edge
    /// (`Some(Visited)`). For a full explanation of each edge type, see the "Depth-first Search"
    /// chapter in [CLR] or [wikipedia].
    ///
    /// If you want to know *both* nodes linked by each edge, you'll need to modify
    /// `TriColorDepthFirstSearch` to store a `source` node for each `Visited` event.
    ///
    /// [wikipedia]: https://en.wikipedia.org/wiki/Depth-first_search#Output_of_a_depth-first_search
    /// [CLR]: https://en.wikipedia.org/wiki/Introduction_to_Algorithms
    fn node_examined(
        &mut self,
        _node: G::Node,
        _prior_status: Option<NodeStatus>,
    ) -> ControlFlow<Self::BreakVal> {
        ControlFlow::CONTINUE
    }

    /// Called after all nodes reachable from this one have been examined.
    fn node_settled(&mut self, _node: G::Node) -> ControlFlow<Self::BreakVal> {
        ControlFlow::CONTINUE
    }

    /// Behave as if no edges exist from `source` to `target`.
    fn ignore_edge(&mut self, _source: G::Node, _target: G::Node) -> bool {
        false
    }
}

/// This `TriColorVisitor` looks for back edges in a graph, which indicate that a cycle exists.
pub struct CycleDetector;

impl<G> TriColorVisitor<G> for CycleDetector
where
    G: ?Sized + DirectedGraph,
{
    type BreakVal = ();

    fn node_examined(
        &mut self,
        _node: G::Node,
        prior_status: Option<NodeStatus>,
    ) -> ControlFlow<Self::BreakVal> {
        match prior_status {
            Some(NodeStatus::Visited) => ControlFlow::BREAK,
            _ => ControlFlow::CONTINUE,
        }
    }
}
