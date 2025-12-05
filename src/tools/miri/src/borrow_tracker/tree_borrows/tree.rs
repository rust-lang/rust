//! In this file we handle the "Tree" part of Tree Borrows, i.e. all tree
//! traversal functions, optimizations to trim branches, and keeping track of
//! the relative position of the access to each node being updated. This of course
//! also includes the definition of the tree structure.
//!
//! Functions here manipulate permissions but are oblivious to them: as
//! the internals of `Permission` are private, the update process is a black
//! box. All we need to know here are
//! - the fact that updates depend only on the old state, the status of protectors,
//!   and the relative position of the access;
//! - idempotency properties asserted in `perms.rs` (for optimizations)

use std::ops::Range;
use std::{cmp, fmt, mem};

use rustc_abi::Size;
use rustc_data_structures::fx::FxHashSet;
use rustc_span::Span;
use smallvec::SmallVec;

use super::diagnostics::AccessCause;
use super::wildcard::WildcardState;
use crate::borrow_tracker::tree_borrows::Permission;
use crate::borrow_tracker::tree_borrows::diagnostics::{
    self, NodeDebugInfo, TbError, TransitionError, no_valid_exposed_references_error,
};
use crate::borrow_tracker::tree_borrows::foreign_access_skipping::IdempotentForeignAccess;
use crate::borrow_tracker::tree_borrows::perms::PermTransition;
use crate::borrow_tracker::tree_borrows::unimap::{UniIndex, UniKeyMap, UniValMap};
use crate::borrow_tracker::{AccessKind, GlobalState, ProtectorKind};
use crate::*;

mod tests;

/// Data for a reference at single *location*.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct LocationState {
    /// A location is "accessed" when it is child-accessed for the first time (and the initial
    /// retag initializes the location for the range covered by the type), and it then stays
    /// accessed forever.
    /// For accessed locations, "permission" is the current permission. However, for
    /// non-accessed locations, we still need to track the "future initial permission": this will
    /// start out to be `default_initial_perm`, but foreign accesses need to be taken into account.
    /// Crucially however, while transitions to `Disabled` would usually be UB if this location is
    /// protected, that is *not* the case for non-accessed locations. Instead we just have a latent
    /// "future initial permission" of `Disabled`, causing UB only if an access is ever actually
    /// performed.
    /// Note that the tree root is also always accessed, as if the allocation was a write access.
    accessed: bool,
    /// This pointer's current permission / future initial permission.
    permission: Permission,
    /// See `foreign_access_skipping.rs`.
    /// Stores an idempotent foreign access for this location and its children.
    /// For correctness, this must not be too strong, and the recorded idempotent foreign access
    /// of all children must be at least as strong as this. For performance, it should be as strong as possible.
    idempotent_foreign_access: IdempotentForeignAccess,
}

impl LocationState {
    /// Constructs a new initial state. It has neither been accessed, nor been subjected
    /// to any foreign access yet.
    /// The permission is not allowed to be `Unique`.
    /// `sifa` is the (strongest) idempotent foreign access, see `foreign_access_skipping.rs`
    pub fn new_non_accessed(permission: Permission, sifa: IdempotentForeignAccess) -> Self {
        assert!(permission.is_initial() || permission.is_disabled());
        Self { permission, accessed: false, idempotent_foreign_access: sifa }
    }

    /// Constructs a new initial state. It has not yet been subjected
    /// to any foreign access. However, it is already marked as having been accessed.
    /// `sifa` is the (strongest) idempotent foreign access, see `foreign_access_skipping.rs`
    pub fn new_accessed(permission: Permission, sifa: IdempotentForeignAccess) -> Self {
        Self { permission, accessed: true, idempotent_foreign_access: sifa }
    }

    /// Check if the location has been accessed, i.e. if it has
    /// ever been accessed through a child pointer.
    pub fn accessed(&self) -> bool {
        self.accessed
    }

    pub fn permission(&self) -> Permission {
        self.permission
    }

    /// Performs an access on this index and updates node,
    /// perm and wildcard_state to reflect the transition.
    fn perform_transition(
        &mut self,
        idx: UniIndex,
        nodes: &mut UniValMap<Node>,
        wildcard_accesses: &mut UniValMap<WildcardState>,
        access_kind: AccessKind,
        access_cause: AccessCause,
        access_range: Option<AllocRange>,
        relatedness: AccessRelatedness,
        span: Span,
        location_range: Range<u64>,
        protected: bool,
    ) -> Result<(), TransitionError> {
        // Call this function now (i.e. only if we know `relatedness`), which
        // ensures it is only called when `skip_if_known_noop` returns
        // `Recurse`, due to the contract of `traverse_this_parents_children_other`.
        self.record_new_access(access_kind, relatedness);

        let transition = self.perform_access(access_kind, relatedness, protected)?;
        if !transition.is_noop() {
            let node = nodes.get_mut(idx).unwrap();
            // Record the event as part of the history.
            node.debug_info.history.push(diagnostics::Event {
                transition,
                is_foreign: relatedness.is_foreign(),
                access_cause,
                access_range,
                transition_range: location_range,
                span,
            });

            // We need to update the wildcard state, if the permission
            // of an exposed pointer changes.
            if node.is_exposed {
                let access_type = self.permission.strongest_allowed_child_access(protected);
                WildcardState::update_exposure(idx, access_type, nodes, wildcard_accesses);
            }
        }
        Ok(())
    }

    /// Apply the effect of an access to one location, including
    /// - applying `Permission::perform_access` to the inner `Permission`,
    /// - emitting protector UB if the location is accessed,
    /// - updating the accessed status (child accesses produce accessed locations).
    fn perform_access(
        &mut self,
        access_kind: AccessKind,
        rel_pos: AccessRelatedness,
        protected: bool,
    ) -> Result<PermTransition, TransitionError> {
        let old_perm = self.permission;
        let transition = Permission::perform_access(access_kind, rel_pos, old_perm, protected)
            .ok_or(TransitionError::ChildAccessForbidden(old_perm))?;
        self.accessed |= !rel_pos.is_foreign();
        self.permission = transition.applied(old_perm).unwrap();
        // Why do only accessed locations cause protector errors?
        // Consider two mutable references `x`, `y` into disjoint parts of
        // the same allocation. A priori, these may actually both be used to
        // access the entire allocation, as long as only reads occur. However,
        // a write to `y` needs to somehow record that `x` can no longer be used
        // on that location at all. For these non-accessed locations (i.e., locations
        // that haven't been accessed with `x` yet), we track the "future initial state":
        // it defaults to whatever the initial state of the tag is,
        // but the access to `y` moves that "future initial state" of `x` to `Disabled`.
        // However, usually a `Reserved -> Disabled` transition would be UB due to the protector!
        // So clearly protectors shouldn't fire for such "future initial state" transitions.
        //
        // See the test `two_mut_protected_same_alloc` in `tests/pass/tree_borrows/tree-borrows.rs`
        // for an example of safe code that would be UB if we forgot to check `self.accessed`.
        if protected && self.accessed && transition.produces_disabled() {
            return Err(TransitionError::ProtectedDisabled(old_perm));
        }
        Ok(transition)
    }

    /// Like `perform_access`, but ignores the concrete error cause and also uses state-passing
    /// rather than a mutable reference. As such, it returns `Some(x)` if the transition succeeded,
    /// or `None` if there was an error.
    #[cfg(test)]
    fn perform_access_no_fluff(
        mut self,
        access_kind: AccessKind,
        rel_pos: AccessRelatedness,
        protected: bool,
    ) -> Option<Self> {
        match self.perform_access(access_kind, rel_pos, protected) {
            Ok(_) => Some(self),
            Err(_) => None,
        }
    }

    /// Tree traversal optimizations. See `foreign_access_skipping.rs`.
    /// This checks if such a foreign access can be skipped.
    fn skip_if_known_noop(
        &self,
        access_kind: AccessKind,
        rel_pos: AccessRelatedness,
    ) -> ContinueTraversal {
        if rel_pos.is_foreign() {
            let happening_now = IdempotentForeignAccess::from_foreign(access_kind);
            let mut new_access_noop =
                self.idempotent_foreign_access.can_skip_foreign_access(happening_now);
            if self.permission.is_disabled() {
                // A foreign access to a `Disabled` tag will have almost no observable effect.
                // It's a theorem that `Disabled` node have no protected accessed children,
                // and so this foreign access will never trigger any protector.
                // (Intuition: You're either protected accessed, and thus can't become Disabled
                // or you're already Disabled protected, but not accessed, and then can't
                // become accessed since that requires a child access, which Disabled blocks.)
                // Further, the children will never be able to read or write again, since they
                // have a `Disabled` parent. So this only affects diagnostics, such that the
                // blocking write will still be identified directly, just at a different tag.
                new_access_noop = true;
            }
            if self.permission.is_frozen() && access_kind == AccessKind::Read {
                // A foreign read to a `Frozen` tag will have almost no observable effect.
                // It's a theorem that `Frozen` nodes have no `Unique` children, so all children
                // already survive foreign reads. Foreign reads in general have almost no
                // effect, the only further thing they could do is make protected `Reserved`
                // nodes become conflicted, i.e. make them reject child writes for the further
                // duration of their protector. But such a child write is already rejected
                // because this node is frozen. So this only affects diagnostics, but the
                // blocking read will still be identified directly, just at a different tag.
                new_access_noop = true;
            }
            if new_access_noop {
                // Abort traversal if the new access is indeed guaranteed
                // to be noop.
                // No need to update `self.idempotent_foreign_access`,
                // the type of the current streak among nonempty read-only
                // or nonempty with at least one write has not changed.
                ContinueTraversal::SkipSelfAndChildren
            } else {
                // Otherwise propagate this time, and also record the
                // access that just occurred so that we can skip the propagation
                // next time.
                ContinueTraversal::Recurse
            }
        } else {
            // A child access occurred, this breaks the streak of foreign
            // accesses in a row and the sequence since the previous child access
            // is now empty.
            ContinueTraversal::Recurse
        }
    }

    /// Records a new access, so that future access can potentially be skipped
    /// by `skip_if_known_noop`. This must be called on child accesses, and otherwise
    /// shoud be called on foreign accesses for increased performance. It should not be called
    /// when `skip_if_known_noop` indicated skipping, since it then is a no-op.
    /// See `foreign_access_skipping.rs`
    fn record_new_access(&mut self, access_kind: AccessKind, rel_pos: AccessRelatedness) {
        debug_assert!(matches!(
            self.skip_if_known_noop(access_kind, rel_pos),
            ContinueTraversal::Recurse
        ));
        self.idempotent_foreign_access
            .record_new(IdempotentForeignAccess::from_acc_and_rel(access_kind, rel_pos));
    }
}

impl fmt::Display for LocationState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.permission)?;
        if !self.accessed {
            write!(f, "?")?;
        }
        Ok(())
    }
}
/// The state of the full tree for a particular location: for all nodes, the local permissions
/// of that node, and the tracking for wildcard accesses.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LocationTree {
    /// Maps a tag to a perm, with possible lazy initialization.
    ///
    /// NOTE: not all tags registered in `Tree::nodes` are necessarily in all
    /// ranges of `perms`, because `perms` is in part lazily initialized.
    /// Just because `nodes.get(key)` is `Some(_)` does not mean you can safely
    /// `unwrap` any `perm.get(key)`.
    ///
    /// We do uphold the fact that `keys(perms)` is a subset of `keys(nodes)`
    pub perms: UniValMap<LocationState>,
    /// Maps a tag and a location to its wildcard access tracking information,
    /// with possible lazy initialization.
    ///
    /// If this allocation doesn't have any exposed nodes, then this map doesn't get
    /// initialized. This way we only need to allocate the map if we need it.
    ///
    /// NOTE: same guarantees on entry initialization as for `perms`.
    pub wildcard_accesses: UniValMap<WildcardState>,
}
/// Tree structure with both parents and children since we want to be
/// able to traverse the tree efficiently in both directions.
#[derive(Clone, Debug)]
pub struct Tree {
    /// Mapping from tags to keys. The key obtained can then be used in
    /// any of the `UniValMap` relative to this allocation, i.e.
    /// `nodes`, `LocationTree::perms` and `LocationTree::wildcard_accesses`
    /// of the same `Tree`.
    /// The parent-child relationship in `Node` is encoded in terms of these same
    /// keys, so traversing the entire tree needs exactly one access to
    /// `tag_mapping`.
    pub(super) tag_mapping: UniKeyMap<BorTag>,
    /// All nodes of this tree.
    pub(super) nodes: UniValMap<Node>,
    /// Associates with each location its state and wildcard access tracking.
    pub(super) locations: DedupRangeMap<LocationTree>,
    /// The index of the root node.
    pub(super) root: UniIndex,
}

/// A node in the borrow tree. Each node is uniquely identified by a tag via
/// the `nodes` map of `Tree`.
#[derive(Clone, Debug)]
pub(super) struct Node {
    /// The tag of this node.
    pub tag: BorTag,
    /// All tags except the root have a parent tag.
    pub parent: Option<UniIndex>,
    /// If the pointer was reborrowed, it has children.
    // FIXME: bench to compare this to FxHashSet and to other SmallVec sizes
    pub children: SmallVec<[UniIndex; 4]>,
    /// Either `Reserved`,  `Frozen`, or `Disabled`, it is the permission this tag will
    /// lazily be initialized to on the first access.
    /// It is only ever `Disabled` for a tree root, since the root is initialized to `Unique` by
    /// its own separate mechanism.
    default_initial_perm: Permission,
    /// The default initial (strongest) idempotent foreign access.
    /// This participates in the invariant for `LocationState::idempotent_foreign_access`
    /// in cases where there is no location state yet. See `foreign_access_skipping.rs`,
    /// and `LocationState::idempotent_foreign_access` for more information
    default_initial_idempotent_foreign_access: IdempotentForeignAccess,
    /// Whether a wildcard access could happen through this node.
    pub is_exposed: bool,
    /// Some extra information useful only for debugging purposes.
    pub debug_info: NodeDebugInfo,
}

/// Data given to the transition function
struct NodeAppArgs<'visit> {
    /// The index of the current node.
    idx: UniIndex,
    /// Relative position of the access.
    rel_pos: AccessRelatedness,
    /// The node map of this tree.
    nodes: &'visit mut UniValMap<Node>,
    /// The permissions map of this tree.
    loc: &'visit mut LocationTree,
}
/// Internal contents of `Tree` with the minimum of mutable access for
/// For soundness do not modify the children or parent indexes of nodes
/// during traversal.
struct TreeVisitor<'tree> {
    nodes: &'tree mut UniValMap<Node>,
    loc: &'tree mut LocationTree,
}

/// Whether to continue exploring the children recursively or not.
enum ContinueTraversal {
    Recurse,
    SkipSelfAndChildren,
}

#[derive(Clone, Copy)]
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
struct TreeVisitorStack<NodeContinue, NodeApp> {
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
}

impl<NodeContinue, NodeApp, Err> TreeVisitorStack<NodeContinue, NodeApp>
where
    NodeContinue: Fn(&NodeAppArgs<'_>) -> ContinueTraversal,
    NodeApp: Fn(NodeAppArgs<'_>) -> Result<(), Err>,
{
    fn should_continue_at(
        &self,
        this: &mut TreeVisitor<'_>,
        idx: UniIndex,
        rel_pos: AccessRelatedness,
    ) -> ContinueTraversal {
        let args = NodeAppArgs { idx, rel_pos, nodes: this.nodes, loc: this.loc };
        (self.f_continue)(&args)
    }

    fn propagate_at(
        &mut self,
        this: &mut TreeVisitor<'_>,
        idx: UniIndex,
        rel_pos: AccessRelatedness,
    ) -> Result<(), Err> {
        (self.f_propagate)(NodeAppArgs { idx, rel_pos, nodes: this.nodes, loc: this.loc })
    }

    fn go_upwards_from_accessed(
        &mut self,
        this: &mut TreeVisitor<'_>,
        accessed_node: UniIndex,
        visit_children: ChildrenVisitMode,
    ) -> Result<(), Err> {
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
        Ok(())
    }

    fn finish_foreign_accesses(&mut self, this: &mut TreeVisitor<'_>) -> Result<(), Err> {
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
        Self { f_continue, f_propagate, stack: Vec::new() }
    }
}

impl<'tree> TreeVisitor<'tree> {
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
    fn traverse_this_parents_children_other<Err>(
        mut self,
        start_idx: UniIndex,
        f_continue: impl Fn(&NodeAppArgs<'_>) -> ContinueTraversal,
        f_propagate: impl Fn(NodeAppArgs<'_>) -> Result<(), Err>,
    ) -> Result<(), Err> {
        let mut stack = TreeVisitorStack::new(f_continue, f_propagate);
        // Visits the accessed node itself, and all its parents, i.e. all nodes
        // undergoing a child access. Also pushes the children and the other
        // cousin nodes (i.e. all nodes undergoing a foreign access) to the stack
        // to be processed later.
        stack.go_upwards_from_accessed(
            &mut self,
            start_idx,
            ChildrenVisitMode::VisitChildrenOfAccessed,
        )?;
        // Now visit all the foreign nodes we remembered earlier.
        // For this we go bottom-up, but also allow f_continue to skip entire
        // subtrees from being visited if it would be a NOP.
        stack.finish_foreign_accesses(&mut self)
    }

    /// Like `traverse_this_parents_children_other`, but skips the children of `start_idx`.
    fn traverse_nonchildren<Err>(
        mut self,
        start_idx: UniIndex,
        f_continue: impl Fn(&NodeAppArgs<'_>) -> ContinueTraversal,
        f_propagate: impl Fn(NodeAppArgs<'_>) -> Result<(), Err>,
    ) -> Result<(), Err> {
        let mut stack = TreeVisitorStack::new(f_continue, f_propagate);
        // Visits the accessed node itself, and all its parents, i.e. all nodes
        // undergoing a child access. Also pushes the other cousin nodes to the
        // stack, but not the children of the accessed node.
        stack.go_upwards_from_accessed(
            &mut self,
            start_idx,
            ChildrenVisitMode::SkipChildrenOfAccessed,
        )?;
        // Now visit all the foreign nodes we remembered earlier.
        // For this we go bottom-up, but also allow f_continue to skip entire
        // subtrees from being visited if it would be a NOP.
        stack.finish_foreign_accesses(&mut self)
    }
}

impl Tree {
    /// Create a new tree, with only a root pointer.
    pub fn new(root_tag: BorTag, size: Size, span: Span) -> Self {
        // The root has `Disabled` as the default permission,
        // so that any access out of bounds is invalid.
        let root_default_perm = Permission::new_disabled();
        let mut tag_mapping = UniKeyMap::default();
        let root_idx = tag_mapping.insert(root_tag);
        let nodes = {
            let mut nodes = UniValMap::<Node>::default();
            let mut debug_info = NodeDebugInfo::new(root_tag, root_default_perm, span);
            // name the root so that all allocations contain one named pointer
            debug_info.add_name("root of the allocation");
            nodes.insert(
                root_idx,
                Node {
                    tag: root_tag,
                    parent: None,
                    children: SmallVec::default(),
                    default_initial_perm: root_default_perm,
                    // The root may never be skipped, all accesses will be local.
                    default_initial_idempotent_foreign_access: IdempotentForeignAccess::None,
                    is_exposed: false,
                    debug_info,
                },
            );
            nodes
        };
        let locations = {
            let mut perms = UniValMap::default();
            // We manually set it to `Unique` on all in-bounds positions.
            // We also ensure that it is accessed, so that no `Unique` but
            // not yet accessed nodes exist. Essentially, we pretend there
            // was a write that initialized these to `Unique`.
            perms.insert(
                root_idx,
                LocationState::new_accessed(
                    Permission::new_unique(),
                    IdempotentForeignAccess::None,
                ),
            );
            let wildcard_accesses = UniValMap::default();
            DedupRangeMap::new(size, LocationTree { perms, wildcard_accesses })
        };
        Self { root: root_idx, nodes, locations, tag_mapping }
    }
}

impl<'tcx> Tree {
    /// Insert a new tag in the tree.
    ///
    /// `inside_perm` defines the initial permissions for a block of memory starting at
    /// `base_offset`. These may nor may not be already marked as "accessed".
    /// `outside_perm` defines the initial permission for the rest of the allocation.
    /// These are definitely not "accessed".
    pub(super) fn new_child(
        &mut self,
        base_offset: Size,
        parent_tag: BorTag,
        new_tag: BorTag,
        inside_perms: DedupRangeMap<LocationState>,
        outside_perm: Permission,
        protected: bool,
        span: Span,
    ) -> InterpResult<'tcx> {
        let idx = self.tag_mapping.insert(new_tag);
        let parent_idx = self.tag_mapping.get(&parent_tag).unwrap();
        assert!(outside_perm.is_initial());

        let default_strongest_idempotent =
            outside_perm.strongest_idempotent_foreign_access(protected);
        // Create the node
        self.nodes.insert(
            idx,
            Node {
                tag: new_tag,
                parent: Some(parent_idx),
                children: SmallVec::default(),
                default_initial_perm: outside_perm,
                default_initial_idempotent_foreign_access: default_strongest_idempotent,
                is_exposed: false,
                debug_info: NodeDebugInfo::new(new_tag, outside_perm, span),
            },
        );
        let parent_node = self.nodes.get_mut(parent_idx).unwrap();
        // Register new_tag as a child of parent_tag
        parent_node.children.push(idx);

        // We need to know the weakest SIFA for `update_idempotent_foreign_access_after_retag`.
        let mut min_sifa = default_strongest_idempotent;
        for (Range { start, end }, &perm) in
            inside_perms.iter(Size::from_bytes(0), inside_perms.size())
        {
            assert!(perm.permission.is_initial());
            assert_eq!(
                perm.idempotent_foreign_access,
                perm.permission.strongest_idempotent_foreign_access(protected)
            );

            min_sifa = cmp::min(min_sifa, perm.idempotent_foreign_access);
            for (_range, loc) in self
                .locations
                .iter_mut(Size::from_bytes(start) + base_offset, Size::from_bytes(end - start))
            {
                loc.perms.insert(idx, perm);
            }
        }

        // We need to ensure the consistency of the wildcard access tracking data structure.
        // For this, we insert the correct entry for this tag based on its parent, if it exists.
        for (_range, loc) in self.locations.iter_mut_all() {
            if let Some(parent_access) = loc.wildcard_accesses.get(parent_idx) {
                loc.wildcard_accesses.insert(idx, parent_access.for_new_child());
            }
        }

        // Inserting the new perms might have broken the SIFA invariant (see
        // `foreign_access_skipping.rs`) if the SIFA we inserted is weaker than that of some parent.
        // We now weaken the recorded SIFA for our parents, until the invariant is restored. We
        // could weaken them all to `None`, but it is more efficient to compute the SIFA for the new
        // permission statically, and use that. For this we need the *minimum* SIFA (`None` needs
        // more fixup than `Write`).
        self.update_idempotent_foreign_access_after_retag(parent_idx, min_sifa);

        interp_ok(())
    }

    /// Restores the SIFA "children are stronger"/"parents are weaker" invariant after a retag:
    /// reduce the SIFA of `current` and its parents to be no stronger than `strongest_allowed`.
    /// See `foreign_access_skipping.rs` and [`Tree::new_child`].
    fn update_idempotent_foreign_access_after_retag(
        &mut self,
        mut current: UniIndex,
        strongest_allowed: IdempotentForeignAccess,
    ) {
        if strongest_allowed == IdempotentForeignAccess::Write {
            // Nothing is stronger than `Write`.
            return;
        }
        // We walk the tree upwards, until the invariant is restored
        loop {
            let current_node = self.nodes.get_mut(current).unwrap();
            // Call `ensure_no_stronger_than` on all SIFAs for this node: the per-location SIFA, as well
            // as the default SIFA for not-yet-initialized locations.
            // Record whether we did any change; if not, the invariant is restored and we can stop the traversal.
            let mut any_change = false;
            for (_range, loc) in self.locations.iter_mut_all() {
                // Check if this node has a state for this location (or range of locations).
                if let Some(perm) = loc.perms.get_mut(current) {
                    // Update the per-location SIFA, recording if it changed.
                    any_change |=
                        perm.idempotent_foreign_access.ensure_no_stronger_than(strongest_allowed);
                }
            }
            // Now update `default_initial_idempotent_foreign_access`, which stores the default SIFA for not-yet-initialized locations.
            any_change |= current_node
                .default_initial_idempotent_foreign_access
                .ensure_no_stronger_than(strongest_allowed);

            if any_change {
                let Some(next) = self.nodes.get(current).unwrap().parent else {
                    // We have arrived at the root.
                    break;
                };
                current = next;
                continue;
            } else {
                break;
            }
        }
    }

    /// Deallocation requires
    /// - a pointer that permits write accesses
    /// - the absence of Strong Protectors anywhere in the allocation
    pub fn dealloc(
        &mut self,
        prov: ProvenanceExtra,
        access_range: AllocRange,
        global: &GlobalState,
        alloc_id: AllocId, // diagnostics
        span: Span,        // diagnostics
    ) -> InterpResult<'tcx> {
        self.perform_access(
            prov,
            Some((access_range, AccessKind::Write, diagnostics::AccessCause::Dealloc)),
            global,
            alloc_id,
            span,
        )?;

        // The order in which we check if any nodes are invalidated only
        // matters to diagnostics, so we use the root as a default tag.
        let start_idx = match prov {
            ProvenanceExtra::Concrete(tag) => self.tag_mapping.get(&tag).unwrap(),
            ProvenanceExtra::Wildcard => self.root,
        };

        // Check if this breaks any strong protector.
        // (Weak protectors are already handled by `perform_access`.)
        for (loc_range, loc) in self.locations.iter_mut(access_range.start, access_range.size) {
            TreeVisitor { nodes: &mut self.nodes, loc }.traverse_this_parents_children_other(
                start_idx,
                // Visit all children, skipping none.
                |_| ContinueTraversal::Recurse,
                |args: NodeAppArgs<'_>| {
                    let node = args.nodes.get(args.idx).unwrap();
                    let perm = args.loc.perms.entry(args.idx);

                    let perm = perm.get().copied().unwrap_or_else(|| node.default_location_state());
                    if global.borrow().protected_tags.get(&node.tag)
                            == Some(&ProtectorKind::StrongProtector)
                            // Don't check for protector if it is a Cell (see `unsafe_cell_deallocate` in `interior_mutability.rs`).
                            // Related to https://github.com/rust-lang/rust/issues/55005.
                            && !perm.permission.is_cell()
                            // Only trigger UB if the accessed bit is set, i.e. if the protector is actually protecting this offset. See #4579.
                            && perm.accessed
                    {
                        Err(TbError {
                            conflicting_info: &node.debug_info,
                            access_cause: diagnostics::AccessCause::Dealloc,
                            alloc_id,
                            error_offset: loc_range.start,
                            error_kind: TransitionError::ProtectedDealloc,
                            accessed_info: match prov {
                                ProvenanceExtra::Concrete(_) =>
                                    Some(&args.nodes.get(start_idx).unwrap().debug_info),
                                // We don't know from where the access came during a wildcard access.
                                ProvenanceExtra::Wildcard => None,
                            },
                        }
                        .build())
                    } else {
                        Ok(())
                    }
                },
            )?;
        }
        interp_ok(())
    }

    /// Map the per-node and per-location `LocationState::perform_access`
    /// to each location of the first component of `access_range_and_kind`,
    /// on every tag of the allocation.
    ///
    /// If `access_range_and_kind` is `None`, this is interpreted as the special
    /// access that is applied on protector release:
    /// - the access will be applied only to accessed locations of the allocation,
    /// - it will not be visible to children,
    /// - it will be recorded as a `FnExit` diagnostic access
    /// - and it will be a read except if the location is `Unique`, i.e. has been written to,
    ///   in which case it will be a write.
    ///
    /// `LocationState::perform_access` will take care of raising transition
    /// errors and updating the `accessed` status of each location,
    /// this traversal adds to that:
    /// - inserting into the map locations that do not exist yet,
    /// - trimming the traversal,
    /// - recording the history.
    pub fn perform_access(
        &mut self,
        prov: ProvenanceExtra,
        access_range_and_kind: Option<(AllocRange, AccessKind, diagnostics::AccessCause)>,
        global: &GlobalState,
        alloc_id: AllocId, // diagnostics
        span: Span,        // diagnostics
    ) -> InterpResult<'tcx> {
        #[cfg(feature = "expensive-consistency-checks")]
        if matches!(prov, ProvenanceExtra::Wildcard) {
            self.verify_wildcard_consistency(global);
        }
        let source_idx = match prov {
            ProvenanceExtra::Concrete(tag) => Some(self.tag_mapping.get(&tag).unwrap()),
            ProvenanceExtra::Wildcard => None,
        };

        if let Some((access_range, access_kind, access_cause)) = access_range_and_kind {
            // Default branch: this is a "normal" access through a known range.
            // We iterate over affected locations and traverse the tree for each of them.
            for (loc_range, loc) in self.locations.iter_mut(access_range.start, access_range.size) {
                loc.perform_access(
                    self.root,
                    &mut self.nodes,
                    source_idx,
                    loc_range,
                    Some(access_range),
                    access_kind,
                    access_cause,
                    global,
                    alloc_id,
                    span,
                    ChildrenVisitMode::VisitChildrenOfAccessed,
                )?;
            }
        } else {
            // This is a special access through the entire allocation.
            // It actually only affects `accessed` locations, so we need
            // to filter on those before initiating the traversal.
            //
            // In addition this implicit access should not be visible to children,
            // thus the use of `traverse_nonchildren`.
            // See the test case `returned_mut_is_usable` from
            // `tests/pass/tree_borrows/tree-borrows.rs` for an example of
            // why this is important.

            // Wildcard references are never protected. So this can never be
            // called with a wildcard reference.
            let source_idx = source_idx.unwrap();

            for (loc_range, loc) in self.locations.iter_mut_all() {
                // Only visit accessed permissions
                if let Some(p) = loc.perms.get(source_idx)
                    && let Some(access_kind) = p.permission.protector_end_access()
                    && p.accessed
                {
                    let access_cause = diagnostics::AccessCause::FnExit(access_kind);
                    loc.perform_access(
                        self.root,
                        &mut self.nodes,
                        Some(source_idx),
                        loc_range,
                        None,
                        access_kind,
                        access_cause,
                        global,
                        alloc_id,
                        span,
                        ChildrenVisitMode::SkipChildrenOfAccessed,
                    )?;
                }
            }
        }
        interp_ok(())
    }
}

/// Integration with the BorTag garbage collector
impl Tree {
    pub fn remove_unreachable_tags(&mut self, live_tags: &FxHashSet<BorTag>) {
        self.remove_useless_children(self.root, live_tags);
        // Right after the GC runs is a good moment to check if we can
        // merge some adjacent ranges that were made equal by the removal of some
        // tags (this does not necessarily mean that they have identical internal representations,
        // see the `PartialEq` impl for `UniValMap`)
        self.locations.merge_adjacent_thorough();
    }

    /// Checks if a node is useless and should be GC'ed.
    /// A node is useless if it has no children and also the tag is no longer live.
    fn is_useless(&self, idx: UniIndex, live: &FxHashSet<BorTag>) -> bool {
        let node = self.nodes.get(idx).unwrap();
        node.children.is_empty() && !live.contains(&node.tag)
    }

    /// Checks whether a node can be replaced by its only child.
    /// If so, returns the index of said only child.
    /// If not, returns none.
    fn can_be_replaced_by_single_child(
        &self,
        idx: UniIndex,
        live: &FxHashSet<BorTag>,
    ) -> Option<UniIndex> {
        let node = self.nodes.get(idx).unwrap();

        let [child_idx] = node.children[..] else { return None };

        // We never want to replace the root node, as it is also kept in `root_ptr_tags`.
        if live.contains(&node.tag) || node.parent.is_none() {
            return None;
        }
        // Since protected nodes are never GC'd (see `borrow_tracker::FrameExtra::visit_provenance`),
        // we know that `node` is not protected because otherwise `live` would
        // have contained `node.tag`.
        let child = self.nodes.get(child_idx).unwrap();
        // Check that for that one child, `can_be_replaced_by_child` holds for the permission
        // on all locations.
        for (_range, loc) in self.locations.iter_all() {
            let parent_perm = loc
                .perms
                .get(idx)
                .map(|x| x.permission)
                .unwrap_or_else(|| node.default_initial_perm);
            let child_perm = loc
                .perms
                .get(child_idx)
                .map(|x| x.permission)
                .unwrap_or_else(|| child.default_initial_perm);
            if !parent_perm.can_be_replaced_by_child(child_perm) {
                return None;
            }
        }

        Some(child_idx)
    }

    /// Properly removes a node.
    /// The node to be removed should not otherwise be usable. It also
    /// should have no children, but this is not checked, so that nodes
    /// whose children were rotated somewhere else can be deleted without
    /// having to first modify them to clear that array.
    fn remove_useless_node(&mut self, this: UniIndex) {
        // Due to the API of UniMap we must make sure to call
        // `UniValMap::remove` for the key of this node on *all* maps that used it
        // (which are `self.nodes` and every range of `self.rperms`)
        // before we can safely apply `UniKeyMap::remove` to truly remove
        // this tag from the `tag_mapping`.
        let node = self.nodes.remove(this).unwrap();
        for (_range, loc) in self.locations.iter_mut_all() {
            loc.perms.remove(this);
            loc.wildcard_accesses.remove(this);
        }
        self.tag_mapping.remove(&node.tag);
    }

    /// Traverses the entire tree looking for useless tags.
    /// Removes from the tree all useless child nodes of root.
    /// It will not delete the root itself.
    ///
    /// NOTE: This leaves in the middle of the tree tags that are unreachable but have
    /// reachable children. There is a potential for compacting the tree by reassigning
    /// children of dead tags to the nearest live parent, but it must be done with care
    /// not to remove UB.
    ///
    /// Example: Consider the tree `root - parent - child`, with `parent: Frozen` and
    /// `child: Reserved`. This tree can exist. If we blindly delete `parent` and reassign
    /// `child` to be a direct child of `root` then Writes to `child` are now permitted
    /// whereas they were not when `parent` was still there.
    fn remove_useless_children(&mut self, root: UniIndex, live: &FxHashSet<BorTag>) {
        // To avoid stack overflows, we roll our own stack.
        // Each element in the stack consists of the current tag, and the number of the
        // next child to be processed.

        // The other functions are written using the `TreeVisitorStack`, but that does not work here
        // since we need to 1) do a post-traversal and 2) remove nodes from the tree.
        // Since we do a post-traversal (by deleting nodes only after handling all children),
        // we also need to be a bit smarter than "pop node, push all children."
        let mut stack = vec![(root, 0)];
        while let Some((tag, nth_child)) = stack.last_mut() {
            let node = self.nodes.get(*tag).unwrap();
            if *nth_child < node.children.len() {
                // Visit the child by pushing it to the stack.
                // Also increase `nth_child` so that when we come back to the `tag` node, we
                // look at the next child.
                let next_child = node.children[*nth_child];
                *nth_child += 1;
                stack.push((next_child, 0));
                continue;
            } else {
                // We have processed all children of `node`, so now it is time to process `node` itself.
                // First, get the current children of `node`. To appease the borrow checker,
                // we have to temporarily move the list out of the node, and then put the
                // list of remaining children back in.
                let mut children_of_node =
                    mem::take(&mut self.nodes.get_mut(*tag).unwrap().children);
                // Remove all useless children.
                children_of_node.retain_mut(|idx| {
                    if self.is_useless(*idx, live) {
                        // Delete `idx` node everywhere else.
                        self.remove_useless_node(*idx);
                        // And delete it from children_of_node.
                        false
                    } else {
                        if let Some(nextchild) = self.can_be_replaced_by_single_child(*idx, live) {
                            // `nextchild` is our grandchild, and will become our direct child.
                            // Delete the in-between node, `idx`.
                            self.remove_useless_node(*idx);
                            // Set the new child's parent.
                            self.nodes.get_mut(nextchild).unwrap().parent = Some(*tag);
                            // Save the new child in children_of_node.
                            *idx = nextchild;
                        }
                        // retain it
                        true
                    }
                });
                // Put back the now-filtered vector.
                self.nodes.get_mut(*tag).unwrap().children = children_of_node;

                // We are done, the parent can continue.
                stack.pop();
                continue;
            }
        }
    }
}

impl<'tcx> LocationTree {
    /// Performs an access on this location.
    /// * `access_source`: The index, if any, where the access came from.
    /// * `visit_children`: Whether to skip updating the children of `access_source`.
    fn perform_access(
        &mut self,
        root: UniIndex,
        nodes: &mut UniValMap<Node>,
        access_source: Option<UniIndex>,
        loc_range: Range<u64>,
        access_range: Option<AllocRange>,
        access_kind: AccessKind,
        access_cause: diagnostics::AccessCause,
        global: &GlobalState,
        alloc_id: AllocId, // diagnostics
        span: Span,        // diagnostics
        visit_children: ChildrenVisitMode,
    ) -> InterpResult<'tcx> {
        if let Some(idx) = access_source {
            self.perform_normal_access(
                idx,
                nodes,
                loc_range.clone(),
                access_range,
                access_kind,
                access_cause,
                global,
                alloc_id,
                span,
                visit_children,
            )
        } else {
            // `SkipChildrenOfAccessed` only gets set on protector release.
            // Since a wildcard reference are never protected this assert shouldn't fail.
            assert!(matches!(visit_children, ChildrenVisitMode::VisitChildrenOfAccessed));
            self.perform_wildcard_access(
                root,
                nodes,
                loc_range.clone(),
                access_range,
                access_kind,
                access_cause,
                global,
                alloc_id,
                span,
            )
        }
    }

    /// Performs a normal access on the tree containing `access_source`.
    /// * `access_source`: The index of the tag being accessed.
    /// * `visit_children`: Whether to skip the children of `access_source`
    ///   during the access. Used for protector end access.
    fn perform_normal_access(
        &mut self,
        access_source: UniIndex,
        nodes: &mut UniValMap<Node>,
        loc_range: Range<u64>,
        access_range: Option<AllocRange>,
        access_kind: AccessKind,
        access_cause: diagnostics::AccessCause,
        global: &GlobalState,
        alloc_id: AllocId, // diagnostics
        span: Span,        // diagnostics
        visit_children: ChildrenVisitMode,
    ) -> InterpResult<'tcx> {
        // Performs the per-node work:
        // - insert the permission if it does not exist
        // - perform the access
        // - record the transition
        // to which some optimizations are added:
        // - skip the traversal of the children in some cases
        // - do not record noop transitions
        //
        // `perms_range` is only for diagnostics (it is the range of
        // the `RangeMap` on which we are currently working).
        let node_skipper = |args: &NodeAppArgs<'_>| -> ContinueTraversal {
            let node = args.nodes.get(args.idx).unwrap();
            let perm = args.loc.perms.get(args.idx);

            let old_state = perm.copied().unwrap_or_else(|| node.default_location_state());
            old_state.skip_if_known_noop(access_kind, args.rel_pos)
        };
        let node_app = |args: NodeAppArgs<'_>| -> Result<(), _> {
            let node = args.nodes.get_mut(args.idx).unwrap();
            let mut perm = args.loc.perms.entry(args.idx);

            let state = perm.or_insert(node.default_location_state());

            let protected = global.borrow().protected_tags.contains_key(&node.tag);
            state
                .perform_transition(
                    args.idx,
                    args.nodes,
                    &mut args.loc.wildcard_accesses,
                    access_kind,
                    access_cause,
                    /* access_range */ access_range,
                    args.rel_pos,
                    span,
                    loc_range.clone(),
                    protected,
                )
                .map_err(|error_kind| {
                    TbError {
                        conflicting_info: &args.nodes.get(args.idx).unwrap().debug_info,
                        access_cause,
                        alloc_id,
                        error_offset: loc_range.start,
                        error_kind,
                        accessed_info: Some(&args.nodes.get(access_source).unwrap().debug_info),
                    }
                    .build()
                })
        };
        let visitor = TreeVisitor { nodes, loc: self };
        match visit_children {
            ChildrenVisitMode::VisitChildrenOfAccessed =>
                visitor.traverse_this_parents_children_other(access_source, node_skipper, node_app),
            ChildrenVisitMode::SkipChildrenOfAccessed =>
                visitor.traverse_nonchildren(access_source, node_skipper, node_app),
        }
        .into()
    }
    /// Performs a wildcard access on the tree with root `root`. Takes the `access_relatedness`
    /// for each node from the `WildcardState` datastructure.
    /// * `root`: Root of the tree being accessed.
    fn perform_wildcard_access(
        &mut self,
        root: UniIndex,
        nodes: &mut UniValMap<Node>,
        loc_range: Range<u64>,
        access_range: Option<AllocRange>,
        access_kind: AccessKind,
        access_cause: diagnostics::AccessCause,
        global: &GlobalState,
        alloc_id: AllocId, // diagnostics
        span: Span,        // diagnostics
    ) -> InterpResult<'tcx> {
        let f_continue =
            |idx: UniIndex, nodes: &UniValMap<Node>, loc: &LocationTree| -> ContinueTraversal {
                let node = nodes.get(idx).unwrap();
                let perm = loc.perms.get(idx);
                let wildcard_state = loc.wildcard_accesses.get(idx).cloned().unwrap_or_default();

                let old_state = perm.copied().unwrap_or_else(|| node.default_location_state());
                // If we know where, relative to this node, the wildcard access occurs,
                // then check if we can skip the entire subtree.
                if let Some(relatedness) = wildcard_state.access_relatedness(access_kind)
                    && let Some(relatedness) = relatedness.to_relatedness()
                {
                    // We can use the usual SIFA machinery to skip nodes.
                    old_state.skip_if_known_noop(access_kind, relatedness)
                } else {
                    ContinueTraversal::Recurse
                }
            };
        // This does a traversal starting from the root through the tree updating
        // the permissions of each node.
        // The difference to `perform_access` is that we take the access
        // relatedness from the wildcard tracking state of the node instead of
        // from the visitor itself.
        TreeVisitor { loc: self, nodes }
            .traverse_this_parents_children_other(
                root,
                |args| f_continue(args.idx, args.nodes, args.loc),
                |args| {
                    let node = args.nodes.get_mut(args.idx).unwrap();
                    let mut entry = args.loc.perms.entry(args.idx);
                    let perm = entry.or_insert(node.default_location_state());

                    let protected = global.borrow().protected_tags.contains_key(&node.tag);

                    let Some(wildcard_relatedness) = args
                        .loc
                        .wildcard_accesses
                        .get(args.idx)
                        .and_then(|s| s.access_relatedness(access_kind))
                    else {
                        // There doesn't exist a valid exposed reference for this access to
                        // happen through.
                        // If this fails for one id, then it fails for all ids so this.
                        // Since we always check the root first, this means it should always
                        // fail on the root.
                        assert_eq!(root, args.idx);
                        return Err(no_valid_exposed_references_error(
                            alloc_id,
                            loc_range.start,
                            access_cause,
                        ));
                    };

                    let Some(relatedness) = wildcard_relatedness.to_relatedness() else {
                        // If the access type is Either, then we do not apply any transition
                        // to this node, but we still update each of its children.
                        // This is an imprecision! In the future, maybe we can still do some sort
                        // of best-effort update here.
                        return Ok(());
                    };
                    // We know the exact relatedness, so we can actually do precise checks.
                    perm.perform_transition(
                        args.idx,
                        args.nodes,
                        &mut args.loc.wildcard_accesses,
                        access_kind,
                        access_cause,
                        access_range,
                        relatedness,
                        span,
                        loc_range.clone(),
                        protected,
                    )
                    .map_err(|trans| {
                        let node = args.nodes.get(args.idx).unwrap();
                        TbError {
                            conflicting_info: &node.debug_info,
                            access_cause,
                            alloc_id,
                            error_offset: loc_range.start,
                            error_kind: trans,
                            // We don't know from where the access came during a wildcard access.
                            accessed_info: None,
                        }
                        .build()
                    })
                },
            )
            .into()
    }
}

impl Node {
    pub fn default_location_state(&self) -> LocationState {
        LocationState::new_non_accessed(
            self.default_initial_perm,
            self.default_initial_idempotent_foreign_access,
        )
    }
}

impl VisitProvenance for Tree {
    fn visit_provenance(&self, visit: &mut VisitWith<'_>) {
        // To ensure that the root never gets removed, we visit it
        // (the `root` node of `Tree` is not an `Option<_>`)
        visit(None, Some(self.nodes.get(self.root).unwrap().tag));

        // We also need to keep around any exposed tags through which
        // an access could still happen.
        for (_id, node) in self.nodes.iter() {
            if node.is_exposed {
                visit(None, Some(node.tag))
            }
        }
    }
}

/// Relative position of the access
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AccessRelatedness {
    /// The access happened either through the node itself or one of
    /// its transitive children.
    LocalAccess,
    /// The access happened through this nodes ancestor or through
    /// a sibling/cousin/uncle/etc.
    ForeignAccess,
}

impl AccessRelatedness {
    /// Check that access is either Ancestor or Distant, i.e. not
    /// a transitive child (initial pointer included).
    pub fn is_foreign(self) -> bool {
        matches!(self, AccessRelatedness::ForeignAccess)
    }
}
