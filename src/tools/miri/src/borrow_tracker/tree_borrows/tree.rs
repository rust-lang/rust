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

use std::fmt;

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

mod tests;

/// Data for a single *location*.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

    /// Check if the location has been initialized, i.e. if it has
    /// ever been accessed through a child pointer.
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Check if the state can exist as the initial permission of a pointer.
    ///
    /// Do not confuse with `is_initialized`, the two are almost orthogonal
    /// as apart from `Active` which is not initial and must be initialized,
    /// any other permission can have an arbitrary combination of being
    /// initial/initialized.
    /// FIXME: when the corresponding `assert` in `tree_borrows/mod.rs` finally
    /// passes and can be uncommented, remove this `#[allow(dead_code)]`.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn is_initial(&self) -> bool {
        self.permission.is_initial()
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
    // `foreign_read_is_noop_after_foreign_write` and `all_transitions_idempotent`,
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
                // (1) justified by `foreign_read_is_noop_after_foreign_write`
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

impl fmt::Display for LocationState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.permission)?;
        if !self.initialized {
            write!(f, "?")?;
        }
        Ok(())
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

/// Stack of nodes left to explore in a tree traversal.
struct TreeVisitorStack<NodeApp, ErrHandler> {
    /// Identifier of the original access.
    initial: UniIndex,
    /// Function to apply to each tag.
    f_propagate: NodeApp,
    /// Handler to add the required context to diagnostics.
    err_builder: ErrHandler,
    /// Mutable state of the visit: the tags left to handle.
    /// Every tag pushed should eventually be handled,
    /// and the precise order is relevant for diagnostics.
    stack: Vec<(UniIndex, AccessRelatedness)>,
}

impl<NodeApp, InnErr, OutErr, ErrHandler> TreeVisitorStack<NodeApp, ErrHandler>
where
    NodeApp: Fn(NodeAppArgs<'_>) -> Result<ContinueTraversal, InnErr>,
    ErrHandler: Fn(ErrHandlerArgs<'_, InnErr>) -> OutErr,
{
    /// Apply the function to the current `tag`, and push its children
    /// to the stack of future tags to visit.
    fn exec_and_visit(
        &mut self,
        this: &mut TreeVisitor<'_>,
        idx: UniIndex,
        exclude: Option<UniIndex>,
        rel_pos: AccessRelatedness,
    ) -> Result<(), OutErr> {
        // 1. apply the propagation function
        let node = this.nodes.get_mut(idx).unwrap();
        let recurse =
            (self.f_propagate)(NodeAppArgs { node, perm: this.perms.entry(idx), rel_pos })
                .map_err(|error_kind| {
                    (self.err_builder)(ErrHandlerArgs {
                        error_kind,
                        conflicting_info: &this.nodes.get(idx).unwrap().debug_info,
                        accessed_info: &this.nodes.get(self.initial).unwrap().debug_info,
                    })
                })?;
        let node = this.nodes.get(idx).unwrap();
        // 2. add the children to the stack for future traversal
        if matches!(recurse, ContinueTraversal::Recurse) {
            let general_child_rel = rel_pos.for_child();
            for &child in node.children.iter() {
                // Some child might be excluded from here and handled separately,
                // e.g. the initially accessed tag.
                if Some(child) != exclude {
                    // We should still ensure that if we don't skip the initially accessed
                    // it will receive the proper `AccessRelatedness`.
                    let this_child_rel = if child == self.initial {
                        AccessRelatedness::This
                    } else {
                        general_child_rel
                    };
                    self.stack.push((child, this_child_rel));
                }
            }
        }
        Ok(())
    }

    fn new(initial: UniIndex, f_propagate: NodeApp, err_builder: ErrHandler) -> Self {
        Self { initial, f_propagate, err_builder, stack: Vec::new() }
    }

    /// Finish the exploration by applying `exec_and_visit` until
    /// the stack is empty.
    fn finish(&mut self, visitor: &mut TreeVisitor<'_>) -> Result<(), OutErr> {
        while let Some((idx, rel_pos)) = self.stack.pop() {
            self.exec_and_visit(visitor, idx, /* no children to exclude */ None, rel_pos)?;
        }
        Ok(())
    }

    /// Push all ancestors to the exploration stack in order of nearest ancestor
    /// towards the top.
    fn push_and_visit_strict_ancestors(
        &mut self,
        visitor: &mut TreeVisitor<'_>,
    ) -> Result<(), OutErr> {
        let mut path_ascend = Vec::new();
        // First climb to the root while recording the path
        let mut curr = self.initial;
        while let Some(ancestor) = visitor.nodes.get(curr).unwrap().parent {
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
            self.exec_and_visit(
                visitor,
                ancestor,
                Some(next_in_path),
                AccessRelatedness::StrictChildAccess,
            )?;
        }
        Ok(())
    }
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
    ) -> Result<(), OutErr> {
        let start_idx = self.tag_mapping.get(&start).unwrap();
        let mut stack = TreeVisitorStack::new(start_idx, f_propagate, err_builder);
        stack.push_and_visit_strict_ancestors(&mut self)?;
        // All (potentially zero) ancestors have been explored,
        // it's time to explore the `start` tag.
        stack.exec_and_visit(
            &mut self,
            start_idx,
            /* no children to exclude */ None,
            AccessRelatedness::This,
        )?;
        // Then finish with a normal DFS.
        stack.finish(&mut self)
    }

    // Applies `f_propagate` to every non-child vertex of the tree (ancestors first).
    //
    // `f_propagate` should follow the following format: for a given `Node` it updates its
    // `Permission` depending on the position relative to `start` (given by an
    // `AccessRelatedness`).
    // It outputs whether the tree traversal for this subree should continue or not.
    fn traverse_nonchildren<InnErr, OutErr>(
        mut self,
        start: BorTag,
        f_propagate: impl Fn(NodeAppArgs<'_>) -> Result<ContinueTraversal, InnErr>,
        err_builder: impl Fn(ErrHandlerArgs<'_, InnErr>) -> OutErr,
    ) -> Result<(), OutErr> {
        let start_idx = self.tag_mapping.get(&start).unwrap();
        let mut stack = TreeVisitorStack::new(start_idx, f_propagate, err_builder);
        stack.push_and_visit_strict_ancestors(&mut self)?;
        // We *don't* visit the `start` tag, and we don't push its children.
        // Only finish the DFS with the cousins.
        stack.finish(&mut self)
    }
}

impl Tree {
    /// Create a new tree, with only a root pointer.
    pub fn new(root_tag: BorTag, size: Size, span: Span) -> Self {
        let root_perm = Permission::new_active();
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
            Some(access_range),
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
    /// If `access_range` is `None`, this is interpreted as the special
    /// access that is applied on protector release:
    /// - the access will be applied only to initialized locations of the allocation,
    /// - and it will not be visible to children.
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
        access_range: Option<AllocRange>,
        global: &GlobalState,
        span: Span,                             // diagnostics
        access_cause: diagnostics::AccessCause, // diagnostics
    ) -> InterpResult<'tcx> {
        use std::ops::Range;
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
        let node_app = |perms_range: Range<u64>,
                        args: NodeAppArgs<'_>|
         -> Result<ContinueTraversal, TransitionError> {
            let NodeAppArgs { node, mut perm, rel_pos } = args;

            let old_state = perm.or_insert(LocationState::new(node.default_initial_perm));

            match old_state.skip_if_known_noop(access_kind, rel_pos) {
                ContinueTraversal::SkipChildren => return Ok(ContinueTraversal::SkipChildren),
                _ => {}
            }

            let protected = global.borrow().protected_tags.contains_key(&node.tag);
            let transition = old_state.perform_access(access_kind, rel_pos, protected)?;

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
        };

        // Error handler in case `node_app` goes wrong.
        // Wraps the faulty transition in more context for diagnostics.
        let err_handler = |perms_range: Range<u64>,
                           args: ErrHandlerArgs<'_, TransitionError>|
         -> InterpError<'tcx> {
            let ErrHandlerArgs { error_kind, conflicting_info, accessed_info } = args;
            TbError {
                conflicting_info,
                access_cause,
                error_offset: perms_range.start,
                error_kind,
                accessed_info,
            }
            .build()
        };

        if let Some(access_range) = access_range {
            // Default branch: this is a "normal" access through a known range.
            // We iterate over affected locations and traverse the tree for each of them.
            for (perms_range, perms) in self.rperms.iter_mut(access_range.start, access_range.size)
            {
                TreeVisitor { nodes: &mut self.nodes, tag_mapping: &self.tag_mapping, perms }
                    .traverse_parents_this_children_others(
                        tag,
                        |args| node_app(perms_range.clone(), args),
                        |args| err_handler(perms_range.clone(), args),
                    )?;
            }
        } else {
            // This is a special access through the entire allocation.
            // It actually only affects `initialized` locations, so we need
            // to filter on those before initiating the traversal.
            //
            // In addition this implicit access should not be visible to children,
            // thus the use of `traverse_nonchildren`.
            // See the test case `returned_mut_is_usable` from
            // `tests/pass/tree_borrows/tree-borrows.rs` for an example of
            // why this is important.
            for (perms_range, perms) in self.rperms.iter_mut_all() {
                let idx = self.tag_mapping.get(&tag).unwrap();
                // Only visit initialized permissions
                if let Some(p) = perms.get(idx)
                    && p.initialized
                {
                    TreeVisitor { nodes: &mut self.nodes, tag_mapping: &self.tag_mapping, perms }
                        .traverse_nonchildren(
                        tag,
                        |args| node_app(perms_range.clone(), args),
                        |args| err_handler(perms_range.clone(), args),
                    )?;
                }
            }
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
