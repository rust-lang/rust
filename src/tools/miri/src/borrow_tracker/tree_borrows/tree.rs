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
use rustc_target::abi::Size;

use crate::borrow_tracker::tree_borrows::{
    diagnostics::{NodeDebugInfo, TbError, TransitionError},
    unimap::{UniEntry, UniIndex, UniKeyMap, UniValMap},
    Permission,
};
use crate::borrow_tracker::{AccessKind, GlobalState, ProtectorKind};
use crate::*;

/// Data for a single *location*.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct LocationState {
    /// This pointer's current permission
    permission: Permission,
    /// A location is initialized when it is child accessed for the first time,
    /// and it then stays initialized forever.
    /// Before initialization we still apply some preemptive transitions on
    /// `permission` to know what to do in case it ever gets initialized,
    /// but these can never cause any immediate UB. There can however be UB
    /// the moment we attempt to initalize (i.e. child-access) because some
    /// foreign access done between the creation and the initialization is
    /// incompatible with child accesses.
    initialized: bool,
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

impl Tree {
    /// Create a new tree, with only a root pointer.
    pub fn new(root_tag: BorTag, size: Size) -> Self {
        let root_perm = Permission::new_root();
        let mut tag_mapping = UniKeyMap::default();
        let root_idx = tag_mapping.insert(root_tag);
        let nodes = {
            let mut nodes = UniValMap::<Node>::default();
            nodes.insert(
                root_idx,
                Node {
                    tag: root_tag,
                    parent: None,
                    children: SmallVec::default(),
                    default_initial_perm: root_perm,
                    debug_info: NodeDebugInfo::new(root_tag),
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
        range: AllocRange,
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
                debug_info: NodeDebugInfo::new(new_tag),
            },
        );
        // Register new_tag as a child of parent_tag
        self.nodes.get_mut(parent_idx).unwrap().children.push(idx);
        // Initialize perms
        let perm = LocationState::new(default_initial_perm).with_access();
        for (_range, perms) in self.rperms.iter_mut(range.start, range.size) {
            perms.insert(idx, perm);
        }
        Ok(())
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
