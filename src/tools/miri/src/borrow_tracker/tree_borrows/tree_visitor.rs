use std::marker::PhantomData;

use super::tree::{AccessRelatedness, Node};
use super::unimap::{UniIndex, UniValMap};

/// Data given to the transition function
pub struct NodeAppArgs<'visit, T> {
    /// The index of the current node.
    pub idx: UniIndex,
    /// Relative position of the access.
    pub rel_pos: AccessRelatedness,
    /// The node map of this tree.
    pub nodes: &'visit mut UniValMap<Node>,
    /// Additional data we want to be able to modify in f_propagate and read in f_continue.
    pub data: &'visit mut T,
}
/// Internal contents of `Tree` with the minimum of mutable access for
/// For soundness do not modify the children or parent indexes of nodes
/// during traversal.
pub struct TreeVisitor<'tree, T> {
    pub nodes: &'tree mut UniValMap<Node>,
    pub data: &'tree mut T,
}

/// Whether to continue exploring the children recursively or not.
#[derive(Debug)]
pub enum ContinueTraversal {
    Recurse,
    SkipSelfAndChildren,
}

#[derive(Clone, Copy, Debug)]
pub enum ChildrenVisitMode {
    VisitChildrenOfAccessed,
    SkipChildrenOfAccessed,
}

enum RecursionState {
    BeforeChildren,
    AfterChildren,
}

/// Stack of nodes left to explore in a tree traversal.
/// See the docs of `traverse_this_parents_children_other` for details on the
/// traversal order.
struct TreeVisitorStack<NodeContinue, NodeApp, T> {
    /// Function describing whether to continue at a tag.
    /// This is only invoked for foreign accesses.
    f_continue: NodeContinue,
    /// Function to apply to each tag.
    f_propagate: NodeApp,
    /// Mutable state of the visit: the tags left to handle.
    /// Every tag pushed should eventually be handled,
    /// and the precise order is relevant for diagnostics.
    /// Since the traversal is piecewise bottom-up, we need to
    /// remember whether we're here initially, or after visiting all children.
    /// The last element indicates this.
    /// This is just an artifact of how you hand-roll recursion,
    /// it does not have a deeper meaning otherwise.
    stack: Vec<(UniIndex, AccessRelatedness, RecursionState)>,
    phantom: PhantomData<T>,
}

impl<NodeContinue, NodeApp, T, Err> TreeVisitorStack<NodeContinue, NodeApp, T>
where
    NodeContinue: Fn(&NodeAppArgs<'_, T>) -> ContinueTraversal,
    NodeApp: FnMut(NodeAppArgs<'_, T>) -> Result<(), Err>,
{
    fn should_continue_at(
        &self,
        this: &mut TreeVisitor<'_, T>,
        idx: UniIndex,
        rel_pos: AccessRelatedness,
    ) -> ContinueTraversal {
        let args = NodeAppArgs { idx, rel_pos, nodes: this.nodes, data: this.data };
        (self.f_continue)(&args)
    }

    fn propagate_at(
        &mut self,
        this: &mut TreeVisitor<'_, T>,
        idx: UniIndex,
        rel_pos: AccessRelatedness,
    ) -> Result<(), Err> {
        (self.f_propagate)(NodeAppArgs { idx, rel_pos, nodes: this.nodes, data: this.data })
    }

    /// Returns the root of this tree.
    fn go_upwards_from_accessed(
        &mut self,
        this: &mut TreeVisitor<'_, T>,
        accessed_node: UniIndex,
        visit_children: ChildrenVisitMode,
    ) -> Result<UniIndex, Err> {
        // We want to visit the accessed node's children first.
        // However, we will below walk up our parents and push their children (our cousins)
        // onto the stack. To ensure correct iteration order, this method thus finishes
        // by reversing the stack. This only works if the stack is empty initially.
        assert!(self.stack.is_empty());
        // First, handle accessed node. A bunch of things need to
        // be handled differently here compared to the further parents
        // of `accesssed_node`.
        {
            self.propagate_at(this, accessed_node, AccessRelatedness::LocalAccess)?;
            if matches!(visit_children, ChildrenVisitMode::VisitChildrenOfAccessed) {
                let accessed_node = this.nodes.get(accessed_node).unwrap();
                // We `rev()` here because we reverse the entire stack later.
                for &child in accessed_node.children.iter().rev() {
                    self.stack.push((
                        child,
                        AccessRelatedness::ForeignAccess,
                        RecursionState::BeforeChildren,
                    ));
                }
            }
        }
        // Then, handle the accessed node's parents. Here, we need to
        // make sure we only mark the "cousin" subtrees for later visitation,
        // not the subtree that contains the accessed node.
        let mut last_node = accessed_node;
        while let Some(current) = this.nodes.get(last_node).unwrap().parent {
            self.propagate_at(this, current, AccessRelatedness::LocalAccess)?;
            let node = this.nodes.get(current).unwrap();
            // We `rev()` here because we reverse the entire stack later.
            for &child in node.children.iter().rev() {
                if last_node == child {
                    continue;
                }
                self.stack.push((
                    child,
                    AccessRelatedness::ForeignAccess,
                    RecursionState::BeforeChildren,
                ));
            }
            last_node = current;
        }
        // Reverse the stack, as discussed above.
        self.stack.reverse();
        Ok(last_node)
    }

    fn finish_foreign_accesses(&mut self, this: &mut TreeVisitor<'_, T>) -> Result<(), Err> {
        while let Some((idx, rel_pos, step)) = self.stack.last_mut() {
            let idx = *idx;
            let rel_pos = *rel_pos;
            match *step {
                // How to do bottom-up traversal, 101: Before you handle a node, you handle all children.
                // For this, you must first find the children, which is what this code here does.
                RecursionState::BeforeChildren => {
                    // Next time we come back will be when all the children are handled.
                    *step = RecursionState::AfterChildren;
                    // Now push the children, except if we are told to skip this subtree.
                    let handle_children = self.should_continue_at(this, idx, rel_pos);
                    match handle_children {
                        ContinueTraversal::Recurse => {
                            let node = this.nodes.get(idx).unwrap();
                            for &child in node.children.iter() {
                                self.stack.push((child, rel_pos, RecursionState::BeforeChildren));
                            }
                        }
                        ContinueTraversal::SkipSelfAndChildren => {
                            // skip self
                            self.stack.pop();
                            continue;
                        }
                    }
                }
                // All the children are handled, let's actually visit this node
                RecursionState::AfterChildren => {
                    self.stack.pop();
                    self.propagate_at(this, idx, rel_pos)?;
                }
            }
        }
        Ok(())
    }

    fn new(f_continue: NodeContinue, f_propagate: NodeApp) -> Self {
        Self { f_continue, f_propagate, stack: Vec::new(), phantom: PhantomData }
    }
}

impl<'tree, T> TreeVisitor<'tree, T> {
    /// Applies `f_propagate` to every vertex of the tree in a piecewise bottom-up way: First, visit
    /// all ancestors of `start_idx` (starting with `start_idx` itself), then children of `start_idx`, then the rest,
    /// going bottom-up in each of these two "pieces" / sections.
    /// This ensures that errors are triggered in the following order
    /// - first invalid accesses with insufficient permissions, closest to the accessed node first,
    /// - then protector violations, bottom-up, starting with the children of the accessed node, and then
    ///   going upwards and outwards.
    ///
    /// The following graphic visualizes it, with numbers indicating visitation order and `start_idx` being
    /// the node that is visited first ("1"):
    ///
    /// ```text
    ///         3
    ///        /|
    ///       / |
    ///      9  2
    ///      |  |\
    ///      |  | \
    ///      8  1  7
    ///        / \
    ///       4   6
    ///           |
    ///           5
    /// ```
    ///
    /// `f_propagate` should follow the following format: for a given `Node` it updates its
    /// `Permission` depending on the position relative to `start_idx` (given by an
    /// `AccessRelatedness`).
    /// `f_continue` is called earlier on foreign nodes, and describes whether to even start
    /// visiting the subtree at that node. If it e.g. returns `SkipSelfAndChildren` on node 6
    /// above, then nodes 5 _and_ 6 would not be visited by `f_propagate`. It is not used for
    /// notes having a child access (nodes 1, 2, 3).
    ///
    /// Finally, remember that the iteration order is not relevant for UB, it only affects
    /// diagnostics. It also affects tree traversal optimizations built on top of this, so
    /// those need to be reviewed carefully as well whenever this changes.
    ///
    /// Returns the index of the root of the accessed tree.
    pub fn traverse_this_parents_children_other<Err>(
        mut self,
        start_idx: UniIndex,
        f_continue: impl Fn(&NodeAppArgs<'_, T>) -> ContinueTraversal,
        f_propagate: impl FnMut(NodeAppArgs<'_, T>) -> Result<(), Err>,
    ) -> Result<UniIndex, Err> {
        let mut stack = TreeVisitorStack::new(f_continue, f_propagate);
        // Visits the accessed node itself, and all its parents, i.e. all nodes
        // undergoing a child access. Also pushes the children and the other
        // cousin nodes (i.e. all nodes undergoing a foreign access) to the stack
        // to be processed later.
        let root = stack.go_upwards_from_accessed(
            &mut self,
            start_idx,
            ChildrenVisitMode::VisitChildrenOfAccessed,
        )?;
        // Now visit all the foreign nodes we remembered earlier.
        // For this we go bottom-up, but also allow f_continue to skip entire
        // subtrees from being visited if it would be a NOP.
        stack.finish_foreign_accesses(&mut self)?;
        Ok(root)
    }

    /// Like `traverse_this_parents_children_other`, but skips the children of `start_idx`.
    ///
    /// Returns the index of the root of the accessed tree.
    pub fn traverse_nonchildren<Err>(
        mut self,
        start_idx: UniIndex,
        f_continue: impl Fn(&NodeAppArgs<'_, T>) -> ContinueTraversal,
        f_propagate: impl FnMut(NodeAppArgs<'_, T>) -> Result<(), Err>,
    ) -> Result<UniIndex, Err> {
        let mut stack = TreeVisitorStack::new(f_continue, f_propagate);
        // Visits the accessed node itself, and all its parents, i.e. all nodes
        // undergoing a child access. Also pushes the other cousin nodes to the
        // stack, but not the children of the accessed node.
        let root = stack.go_upwards_from_accessed(
            &mut self,
            start_idx,
            ChildrenVisitMode::SkipChildrenOfAccessed,
        )?;
        // Now visit all the foreign nodes we remembered earlier.
        // For this we go bottom-up, but also allow f_continue to skip entire
        // subtrees from being visited if it would be a NOP.
        stack.finish_foreign_accesses(&mut self)?;
        Ok(root)
    }

    /// Traverses all children of `start_idx` including `start_idx` itself.
    /// Uses `f_continue` to filter out subtrees and then processes each node
    /// with `f_propagate` so that the children get processed before their
    /// parents.
    pub fn traverse_children_this<Err>(
        mut self,
        start_idx: UniIndex,
        f_continue: impl Fn(&NodeAppArgs<'_, T>) -> ContinueTraversal,
        f_propagate: impl FnMut(NodeAppArgs<'_, T>) -> Result<(), Err>,
    ) -> Result<(), Err> {
        let mut stack = TreeVisitorStack::new(f_continue, f_propagate);

        stack.stack.push((
            start_idx,
            AccessRelatedness::ForeignAccess,
            RecursionState::BeforeChildren,
        ));
        stack.finish_foreign_accesses(&mut self)
    }
}
