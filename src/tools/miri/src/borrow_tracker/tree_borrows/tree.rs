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

use smallvec::SmallVec;

use rustc_const_eval::interpret::InterpResult;
use rustc_data_structures::fx::FxHashSet;
use rustc_span::Span;
use rustc_target::abi::Size;

use crate::borrow_tracker::tree_borrows::{
    diagnostics::{self, NodeDebugInfo, TbError, TransitionError},
    perms::PermTransition,
    unimap::{UniEntry, UniIndex, UniKeyMap, UniValMap},
    Permission,
};
use crate::borrow_tracker::{AccessKind, GlobalState, ProtectorKind};
use crate::*;

/// Data for a single *location*.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct LocationState {
    /// A location is initialized when it is child-accessed for the first time (and the initial
    /// retag initializes the location for the range covered by the type), and it then stays
    /// initialized forever.
    /// For initialized locations, "permission" is the current permission. However, for
    /// uninitialized locations, we still need to track the "future initial permission": this will
    /// start out to be `default_initial_perm`, but foreign accesses need to be taken into account.
    /// Crucially however, while transitions to `Disabled` would usually be UB if this location is
    /// protected, that is *not* the case for uninitialized locations. Instead we just have a latent
    /// "future initial permission" of `Disabled`, causing UB only if an access is ever actually
    /// performed.
    initialized: bool,
    /// This pointer's current permission / future initial permission.
    permission: Permission,
    /// Strongest foreign access whose effects have already been applied to
    /// this node and all its children since the last child access.
    /// This is `None` if the most recent access is a child access,
    /// `Some(Write)` if at least one foreign write access has been applied
    /// since the previous child access, and `Some(Read)` if at least one
    /// foreign read and no foreign write have occurred since the last child access.
    latest_foreign_access: Option<AccessKind>,
}

impl LocationState {
    /// Default initial state has never been accessed and has been subjected to no
    /// foreign access.
    fn new(permission: Permission) -> Self {
        Self { permission, initialized: false, latest_foreign_access: None }
    }

    /// Record that this location was accessed through a child pointer by
    /// marking it as initialized
    fn with_access(mut self) -> Self {
        self.initialized = true;
        self
    }

    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    pub fn permission(&self) -> Permission {
        self.permission
    }

    /// Apply the effect of an access to one location, including
    /// - applying `Permission::perform_access` to the inner `Permission`,
    /// - emitting protector UB if the location is initialized,
    /// - updating the initialized status (child accesses produce initialized locations).
    fn perform_access(
        &mut self,
        access_kind: AccessKind,
        rel_pos: AccessRelatedness,
        protected: bool,
    ) -> Result<PermTransition, TransitionError> {
        let old_perm = self.permission;
        let transition = Permission::perform_access(access_kind, rel_pos, old_perm, protected)
            .ok_or(TransitionError::ChildAccessForbidden(old_perm))?;
        // Why do only initialized locations cause protector errors?
        // Consider two mutable references `x`, `y` into disjoint parts of
        // the same allocation. A priori, these may actually both be used to
        // access the entire allocation, as long as only reads occur. However,
        // a write to `y` needs to somehow record that `x` can no longer be used
        // on that location at all. For these uninitialized locations (i.e., locations
        // that haven't been accessed with `x` yet), we track the "future initial state":
        // it defaults to whatever the initial state of the tag is,
        // but the access to `y` moves that "future initial state" of `x` to `Disabled`.
        // However, usually a `Reserved -> Disabled` transition would be UB due to the protector!
        // So clearly protectors shouldn't fire for such "future initial state" transitions.
        //
        // See the test `two_mut_protected_same_alloc` in `tests/pass/tree_borrows/tree-borrows.rs`
        // for an example of safe code that would be UB if we forgot to check `self.initialized`.
        if protected && self.initialized && transition.produces_disabled() {
            return Err(TransitionError::ProtectedDisabled(old_perm));
        }
        self.permission = transition.applied(old_perm).unwrap();
        self.initialized |= !rel_pos.is_foreign();
        Ok(transition)
    }

    // Helper to optimize the tree traversal.
    // The optimization here consists of observing thanks to the tests
    // `foreign_read_is_noop_after_write` and `all_transitions_idempotent`,
    // that there are actually just three possible sequences of events that can occur
    // in between two child accesses that produce different results.
    //
    // Indeed,
    // - applying any number of foreign read accesses is the same as applying
    //   exactly one foreign read,
    // - applying any number of foreign read or write accesses is the same
    //   as applying exactly one foreign write.
    // therefore the three sequences of events that can produce different
    // outcomes are
    // - an empty sequence (`self.latest_foreign_access = None`)
    // - a nonempty read-only sequence (`self.latest_foreign_access = Some(Read)`)
    // - a nonempty sequence with at least one write (`self.latest_foreign_access = Some(Write)`)
    //
    // This function not only determines if skipping the propagation right now
    // is possible, it also updates the internal state to keep track of whether
    // the propagation can be skipped next time.
    // It is a performance loss not to call this function when a foreign access occurs.
    // It is unsound not to call this function when a child access occurs.
    fn skip_if_known_noop(
        &mut self,
        access_kind: AccessKind,
        rel_pos: AccessRelatedness,
    ) -> ContinueTraversal {
        if rel_pos.is_foreign() {
            let new_access_noop = match (self.latest_foreign_access, access_kind) {
                // Previously applied transition makes the new one a guaranteed
                // noop in the two following cases:
                // (1) justified by `foreign_read_is_noop_after_write`
                (Some(AccessKind::Write), AccessKind::Read) => true,
                // (2) justified by `all_transitions_idempotent`
                (Some(old), new) if old == new => true,
                // In all other cases there has been a recent enough
                // child access that the effects of the new foreign access
                // need to be applied to this subtree.
                _ => false,
            };
            if new_access_noop {
                // Abort traversal if the new transition is indeed guaranteed
                // to be noop.
                // No need to update `self.latest_foreign_access`,
                // the type of the current streak among nonempty read-only
                // or nonempty with at least one write has not changed.
                ContinueTraversal::SkipChildren
            } else {
                // Otherwise propagate this time, and also record the
                // access that just occurred so that we can skip the propagation
                // next time.
                self.latest_foreign_access = Some(access_kind);
                ContinueTraversal::Recurse
            }
        } else {
            // A child access occurred, this breaks the streak of foreign
            // accesses in a row and the sequence since the previous child access
            // is now empty.
            self.latest_foreign_access = None;
            ContinueTraversal::Recurse
        }
    }
}

/// Tree structure with both parents and children since we want to be
/// able to traverse the tree efficiently in both directions.
#[derive(Clone, Debug)]
pub struct Tree {
    /// Mapping from tags to keys. The key obtained can then be used in
    /// any of the `UniValMap` relative to this allocation, i.e. both the
    /// `nodes` and `rperms` of the same `Tree`.
    /// The parent-child relationship in `Node` is encoded in terms of these same
    /// keys, so traversing the entire tree needs exactly one access to
    /// `tag_mapping`.
    pub(super) tag_mapping: UniKeyMap<BorTag>,
    /// All nodes of this tree.
    pub(super) nodes: UniValMap<Node>,
    /// Maps a tag and a location to a perm, with possible lazy
    /// initialization.
    ///
    /// NOTE: not all tags registered in `nodes` are necessarily in all
    /// ranges of `rperms`, because `rperms` is in part lazily initialized.
    /// Just because `nodes.get(key)` is `Some(_)` does not mean you can safely
    /// `unwrap` any `perm.get(key)`.
    ///
    /// We do uphold the fact that `keys(perms)` is a subset of `keys(nodes)`
    pub(super) rperms: RangeMap<UniValMap<LocationState>>,
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
    /// Either `Reserved` or `Frozen`, the permission this tag will be lazily initialized
    /// to on the first access.
    default_initial_perm: Permission,
    /// Some extra information useful only for debugging purposes
    pub debug_info: NodeDebugInfo,
}

/// Data given to the transition function
struct NodeAppArgs<'node> {
    /// Node on which the transition is currently being applied
    node: &'node mut Node,
    /// Mutable access to its permissions
    perm: UniEntry<'node, LocationState>,
    /// Relative position of the access
    rel_pos: AccessRelatedness,
}
/// Data given to the error handler
struct ErrHandlerArgs<'node, InErr> {
    /// Kind of error that occurred
    error_kind: InErr,
    /// Tag that triggered the error (not the tag that was accessed,
    /// rather the parent tag that had insufficient permissions or the
    /// non-parent tag that had a protector).
    conflicting_info: &'node NodeDebugInfo,
    /// Information about the tag that was accessed just before the
    /// error was triggered.
    accessed_info: &'node NodeDebugInfo,
}
/// Internal contents of `Tree` with the minimum of mutable access for
/// the purposes of the tree traversal functions: the permissions (`perms`) can be
/// updated but not the tree structure (`tag_mapping` and `nodes`)
struct TreeVisitor<'tree> {
    tag_mapping: &'tree UniKeyMap<BorTag>,
    nodes: &'tree mut UniValMap<Node>,
    perms: &'tree mut UniValMap<LocationState>,
}

/// Whether to continue exploring the children recursively or not.
enum ContinueTraversal {
    Recurse,
    SkipChildren,
}

impl<'tree> TreeVisitor<'tree> {
    // Applies `f_propagate` to every vertex of the tree top-down in the following order: first
    // all ancestors of `start`, then `start` itself, then children of `start`, then the rest.
    // This ensures that errors are triggered in the following order
    // - first invalid accesses with insufficient permissions, closest to the root first,
    // - then protector violations, closest to `start` first.
    //
    // `f_propagate` should follow the following format: for a given `Node` it updates its
    // `Permission` depending on the position relative to `start` (given by an
    // `AccessRelatedness`).
    // It outputs whether the tree traversal for this subree should continue or not.
    fn traverse_parents_this_children_others<InnErr, OutErr>(
        mut self,
        start: BorTag,
        f_propagate: impl Fn(NodeAppArgs<'_>) -> Result<ContinueTraversal, InnErr>,
        err_builder: impl Fn(ErrHandlerArgs<'_, InnErr>) -> OutErr,
    ) -> Result<(), OutErr>
where {
        struct TreeVisitAux<NodeApp, ErrHandler> {
            accessed_tag: UniIndex,
            f_propagate: NodeApp,
            err_builder: ErrHandler,
            stack: Vec<(UniIndex, AccessRelatedness)>,
        }
        impl<NodeApp, InnErr, OutErr, ErrHandler> TreeVisitAux<NodeApp, ErrHandler>
        where
            NodeApp: Fn(NodeAppArgs<'_>) -> Result<ContinueTraversal, InnErr>,
            ErrHandler: Fn(ErrHandlerArgs<'_, InnErr>) -> OutErr,
        {
            fn pop(&mut self) -> Option<(UniIndex, AccessRelatedness)> {
                self.stack.pop()
            }

            /// Apply the function to the current `tag`, and push its children
            /// to the stack of future tags to visit.
            fn exec_and_visit(
                &mut self,
                this: &mut TreeVisitor<'_>,
                tag: UniIndex,
                exclude: Option<UniIndex>,
                rel_pos: AccessRelatedness,
            ) -> Result<(), OutErr> {
                // 1. apply the propagation function
                let node = this.nodes.get_mut(tag).unwrap();
                let recurse =
                    (self.f_propagate)(NodeAppArgs { node, perm: this.perms.entry(tag), rel_pos })
                        .map_err(|error_kind| {
                            (self.err_builder)(ErrHandlerArgs {
                                error_kind,
                                conflicting_info: &this.nodes.get(tag).unwrap().debug_info,
                                accessed_info: &this
                                    .nodes
                                    .get(self.accessed_tag)
                                    .unwrap()
                                    .debug_info,
                            })
                        })?;
                let node = this.nodes.get(tag).unwrap();
                // 2. add the children to the stack for future traversal
                if matches!(recurse, ContinueTraversal::Recurse) {
                    let child_rel = rel_pos.for_child();
                    for &child in node.children.iter() {
                        // some child might be excluded from here and handled separately
                        if Some(child) != exclude {
                            self.stack.push((child, child_rel));
                        }
                    }
                }
                Ok(())
            }
        }

        let start_idx = self.tag_mapping.get(&start).unwrap();
        let mut stack =
            TreeVisitAux { accessed_tag: start_idx, f_propagate, err_builder, stack: Vec::new() };
        {
            let mut path_ascend = Vec::new();
            // First climb to the root while recording the path
            let mut curr = start_idx;
            while let Some(ancestor) = self.nodes.get(curr).unwrap().parent {
                path_ascend.push((ancestor, curr));
                curr = ancestor;
            }
            // Then descend:
            // - execute f_propagate on each node
            // - record children in visit
            while let Some((ancestor, next_in_path)) = path_ascend.pop() {
                // Explore ancestors in descending order.
                // `next_in_path` is excluded from the recursion because it
                // will be the `ancestor` of the next iteration.
                // It also needs a different `AccessRelatedness` than the other
                // children of `ancestor`.
                stack.exec_and_visit(
                    &mut self,
                    ancestor,
                    Some(next_in_path),
                    AccessRelatedness::StrictChildAccess,
                )?;
            }
        };
        // All (potentially zero) ancestors have been explored, call f_propagate on start
        stack.exec_and_visit(&mut self, start_idx, None, AccessRelatedness::This)?;
        // up to this point we have never popped from `stack`, hence if the
        // path to the root is `root = p(n) <- p(n-1)... <- p(1) <- p(0) = start`
        // then now `stack` contains
        // `[<children(p(n)) except p(n-1)> ... <children(p(1)) except p(0)> <children(p(0))>]`,
        // all of which are for now unexplored.
        // This is the starting point of a standard DFS which will thus
        // explore all non-ancestors of `start` in the following order:
        // - all descendants of `start`;
        // - then the unexplored descendants of `parent(start)`;
        // ...
        // - until finally the unexplored descendants of `root`.
        while let Some((tag, rel_pos)) = stack.pop() {
            stack.exec_and_visit(&mut self, tag, None, rel_pos)?;
        }
        Ok(())
    }
}

impl Tree {
    /// Create a new tree, with only a root pointer.
    pub fn new(root_tag: BorTag, size: Size, span: Span) -> Self {
        let root_perm = Permission::new_root();
        let mut tag_mapping = UniKeyMap::default();
        let root_idx = tag_mapping.insert(root_tag);
        let nodes = {
            let mut nodes = UniValMap::<Node>::default();
            let mut debug_info = NodeDebugInfo::new(root_tag, root_perm, span);
            // name the root so that all allocations contain one named pointer
            debug_info.add_name("root of the allocation");
            nodes.insert(
                root_idx,
                Node {
                    tag: root_tag,
                    parent: None,
                    children: SmallVec::default(),
                    default_initial_perm: root_perm,
                    debug_info,
                },
            );
            nodes
        };
        let rperms = {
            let mut perms = UniValMap::default();
            perms.insert(root_idx, LocationState::new(root_perm).with_access());
            RangeMap::new(size, perms)
        };
        Self { root: root_idx, nodes, rperms, tag_mapping }
    }
}

impl<'tcx> Tree {
    /// Insert a new tag in the tree
    pub fn new_child(
        &mut self,
        parent_tag: BorTag,
        new_tag: BorTag,
        default_initial_perm: Permission,
        reborrow_range: AllocRange,
        span: Span,
    ) -> InterpResult<'tcx> {
        assert!(!self.tag_mapping.contains_key(&new_tag));
        let idx = self.tag_mapping.insert(new_tag);
        let parent_idx = self.tag_mapping.get(&parent_tag).unwrap();
        // Create the node
        self.nodes.insert(
            idx,
            Node {
                tag: new_tag,
                parent: Some(parent_idx),
                children: SmallVec::default(),
                default_initial_perm,
                debug_info: NodeDebugInfo::new(new_tag, default_initial_perm, span),
            },
        );
        // Register new_tag as a child of parent_tag
        self.nodes.get_mut(parent_idx).unwrap().children.push(idx);
        // Initialize perms
        let perm = LocationState::new(default_initial_perm).with_access();
        for (_perms_range, perms) in self.rperms.iter_mut(reborrow_range.start, reborrow_range.size)
        {
            perms.insert(idx, perm);
        }
        Ok(())
    }

    /// Deallocation requires
    /// - a pointer that permits write accesses
    /// - the absence of Strong Protectors anywhere in the allocation
    pub fn dealloc(
        &mut self,
        tag: BorTag,
        access_range: AllocRange,
        global: &GlobalState,
        span: Span, // diagnostics
    ) -> InterpResult<'tcx> {
        self.perform_access(
            AccessKind::Write,
            tag,
            access_range,
            global,
            span,
            diagnostics::AccessCause::Dealloc,
        )?;
        for (perms_range, perms) in self.rperms.iter_mut(access_range.start, access_range.size) {
            TreeVisitor { nodes: &mut self.nodes, tag_mapping: &self.tag_mapping, perms }
                .traverse_parents_this_children_others(
                    tag,
                    |args: NodeAppArgs<'_>| -> Result<ContinueTraversal, TransitionError> {
                        let NodeAppArgs { node, .. } = args;
                        if global.borrow().protected_tags.get(&node.tag)
                            == Some(&ProtectorKind::StrongProtector)
                        {
                            Err(TransitionError::ProtectedDealloc)
                        } else {
                            Ok(ContinueTraversal::Recurse)
                        }
                    },
                    |args: ErrHandlerArgs<'_, TransitionError>| -> InterpError<'tcx> {
                        let ErrHandlerArgs { error_kind, conflicting_info, accessed_info } = args;
                        TbError {
                            conflicting_info,
                            access_cause: diagnostics::AccessCause::Dealloc,
                            error_offset: perms_range.start,
                            error_kind,
                            accessed_info,
                        }
                        .build()
                    },
                )?;
        }
        Ok(())
    }

    /// Map the per-node and per-location `LocationState::perform_access`
    /// to each location of `access_range`, on every tag of the allocation.
    ///
    /// `LocationState::perform_access` will take care of raising transition
    /// errors and updating the `initialized` status of each location,
    /// this traversal adds to that:
    /// - inserting into the map locations that do not exist yet,
    /// - trimming the traversal,
    /// - recording the history.
    pub fn perform_access(
        &mut self,
        access_kind: AccessKind,
        tag: BorTag,
        access_range: AllocRange,
        global: &GlobalState,
        span: Span,                             // diagnostics
        access_cause: diagnostics::AccessCause, // diagnostics
    ) -> InterpResult<'tcx> {
        for (perms_range, perms) in self.rperms.iter_mut(access_range.start, access_range.size) {
            TreeVisitor { nodes: &mut self.nodes, tag_mapping: &self.tag_mapping, perms }
                .traverse_parents_this_children_others(
                    tag,
                    |args: NodeAppArgs<'_>| -> Result<ContinueTraversal, TransitionError> {
                        let NodeAppArgs { node, mut perm, rel_pos } = args;

                        let old_state =
                            perm.or_insert_with(|| LocationState::new(node.default_initial_perm));

                        match old_state.skip_if_known_noop(access_kind, rel_pos) {
                            ContinueTraversal::SkipChildren =>
                                return Ok(ContinueTraversal::SkipChildren),
                            _ => {}
                        }

                        let protected = global.borrow().protected_tags.contains_key(&node.tag);
                        let transition =
                            old_state.perform_access(access_kind, rel_pos, protected)?;

                        // Record the event as part of the history
                        if !transition.is_noop() {
                            node.debug_info.history.push(diagnostics::Event {
                                transition,
                                is_foreign: rel_pos.is_foreign(),
                                access_cause,
                                access_range,
                                transition_range: perms_range.clone(),
                                span,
                            });
                        }
                        Ok(ContinueTraversal::Recurse)
                    },
                    |args: ErrHandlerArgs<'_, TransitionError>| -> InterpError<'tcx> {
                        let ErrHandlerArgs { error_kind, conflicting_info, accessed_info } = args;
                        TbError {
                            conflicting_info,
                            access_cause,
                            error_offset: perms_range.start,
                            error_kind,
                            accessed_info,
                        }
                        .build()
                    },
                )?;
        }
        Ok(())
    }
}

/// Integration with the BorTag garbage collector
impl Tree {
    pub fn remove_unreachable_tags(&mut self, live_tags: &FxHashSet<BorTag>) {
        let root_is_needed = self.keep_only_needed(self.root, live_tags); // root can't be removed
        assert!(root_is_needed);
        // Right after the GC runs is a good moment to check if we can
        // merge some adjacent ranges that were made equal by the removal of some
        // tags (this does not necessarily mean that they have identical internal representations,
        // see the `PartialEq` impl for `UniValMap`)
        self.rperms.merge_adjacent_thorough();
    }

    /// Traverses the entire tree looking for useless tags.
    /// Returns true iff the tag it was called on is still live or has live children,
    /// and removes from the tree all tags that have no live children.
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
    fn keep_only_needed(&mut self, idx: UniIndex, live: &FxHashSet<BorTag>) -> bool {
        let node = self.nodes.get(idx).unwrap();
        // FIXME: this function does a lot of cloning, a 2-pass approach is possibly
        // more efficient. It could consist of
        // 1. traverse the Tree, collect all useless tags in a Vec
        // 2. traverse the Vec, remove all tags previously selected
        // Bench it.
        let children: SmallVec<_> = node
            .children
            .clone()
            .into_iter()
            .filter(|child| self.keep_only_needed(*child, live))
            .collect();
        let no_children = children.is_empty();
        let node = self.nodes.get_mut(idx).unwrap();
        node.children = children;
        if !live.contains(&node.tag) && no_children {
            // All of the children and this node are unreachable, delete this tag
            // from the tree (the children have already been deleted by recursive
            // calls).
            // Due to the API of UniMap we must absolutely call
            // `UniValMap::remove` for the key of this tag on *all* maps that used it
            // (which are `self.nodes` and every range of `self.rperms`)
            // before we can safely apply `UniValMap::forget` to truly remove
            // the tag from the mapping.
            let tag = node.tag;
            self.nodes.remove(idx);
            for (_perms_range, perms) in self.rperms.iter_mut_all() {
                perms.remove(idx);
            }
            self.tag_mapping.remove(&tag);
            // The tag has been deleted, inform the caller
            false
        } else {
            // The tag is still live or has live children, it must be kept
            true
        }
    }
}

impl VisitTags for Tree {
    fn visit_tags(&self, visit: &mut dyn FnMut(BorTag)) {
        // To ensure that the root never gets removed, we visit it
        // (the `root` node of `Tree` is not an `Option<_>`)
        visit(self.nodes.get(self.root).unwrap().tag)
    }
}

/// Relative position of the access
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AccessRelatedness {
    /// The accessed pointer is the current one
    This,
    /// The accessed pointer is a (transitive) child of the current one.
    // Current pointer is excluded (unlike in some other places of this module
    // where "child" is inclusive).
    StrictChildAccess,
    /// The accessed pointer is a (transitive) parent of the current one.
    // Current pointer is excluded.
    AncestorAccess,
    /// The accessed pointer is neither of the above.
    // It's a cousin/uncle/etc., something in a side branch.
    // FIXME: find a better name ?
    DistantAccess,
}

impl AccessRelatedness {
    /// Check that access is either Ancestor or Distant, i.e. not
    /// a transitive child (initial pointer included).
    pub fn is_foreign(self) -> bool {
        matches!(self, AccessRelatedness::AncestorAccess | AccessRelatedness::DistantAccess)
    }

    /// Given the AccessRelatedness for the parent node, compute the AccessRelatedness
    /// for the child node. This function assumes that we propagate away from the initial
    /// access.
    pub fn for_child(self) -> Self {
        use AccessRelatedness::*;
        match self {
            AncestorAccess | This => AncestorAccess,
            StrictChildAccess | DistantAccess => DistantAccess,
        }
    }
}

#[cfg(test)]
mod commutation_tests {
    use super::*;
    impl LocationState {
        pub fn all_without_access() -> impl Iterator<Item = Self> {
            Permission::all().flat_map(|permission| {
                [false, true].into_iter().map(move |initialized| {
                    Self { permission, initialized, latest_foreign_access: None }
                })
            })
        }
    }

    #[test]
    #[rustfmt::skip]
    // Exhaustive check that for any starting configuration loc,
    // for any two read accesses r1 and r2, if `loc + r1 + r2` is not UB
    // and results in `loc'`, then `loc + r2 + r1` is also not UB and results
    // in the same final state `loc'`.
    // This lets us justify arbitrary read-read reorderings.
    fn all_read_accesses_commute() {
        let kind = AccessKind::Read;
        // Two of the four combinations of `AccessRelatedness` are trivial,
        // but we might as well check them all.
        for rel1 in AccessRelatedness::all() {
            for rel2 in AccessRelatedness::all() {
                // Any protector state works, but we can't move reads across function boundaries
                // so the two read accesses occur under the same protector.
                for &protected in &[true, false] {
                    for loc in LocationState::all_without_access() {
                        // Apply 1 then 2. Failure here means that there is UB in the source
                        // and we skip the check in the target.
                        let mut loc12 = loc;
                        let Ok(_) = loc12.perform_access(kind, rel1, protected) else { continue; };
                        let Ok(_) = loc12.perform_access(kind, rel2, protected) else { continue; };

                        // If 1 followed by 2 succeeded, then 2 followed by 1 must also succeed...
                        let mut loc21 = loc;
                        loc21.perform_access(kind, rel2, protected).unwrap();
                        loc21.perform_access(kind, rel1, protected).unwrap();

                        // ... and produce the same final result.
                        assert_eq!(
                            loc12, loc21,
                            "Read accesses {:?} followed by {:?} do not commute !",
                            rel1, rel2
                        );
                    }
                }
            }
        }
    }
}
