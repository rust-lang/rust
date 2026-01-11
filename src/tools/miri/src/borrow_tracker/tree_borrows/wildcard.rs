use std::cmp::max;
use std::fmt::Debug;

use super::Tree;
use super::tree::{AccessRelatedness, Node};
use super::unimap::{UniIndex, UniValMap};
use crate::BorTag;
use crate::borrow_tracker::AccessKind;
#[cfg(feature = "expensive-consistency-checks")]
use crate::borrow_tracker::GlobalState;

/// Represents the maximum access level that is possible.
///
/// Note that we derive Ord and PartialOrd, so the order in which variants are listed below matters:
/// None < Read < Write. Do not change that order.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
pub enum WildcardAccessLevel {
    #[default]
    None,
    Read,
    Write,
}
impl WildcardAccessLevel {
    /// Weather this access kind is allowed at this level.
    pub fn allows(self, kind: AccessKind) -> bool {
        let required_level = match kind {
            AccessKind::Read => Self::Read,
            AccessKind::Write => Self::Write,
        };
        required_level <= self
    }
}

/// Where the access happened relative to the current node.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WildcardAccessRelatedness {
    /// The access definitively happened through a local node.
    LocalAccess,
    /// The access definitively happened through a foreign node.
    ForeignAccess,
    /// We do not know if the access is foreign or local.
    EitherAccess,
}
impl WildcardAccessRelatedness {
    pub fn to_relatedness(self) -> Option<AccessRelatedness> {
        match self {
            Self::LocalAccess => Some(AccessRelatedness::LocalAccess),
            Self::ForeignAccess => Some(AccessRelatedness::ForeignAccess),
            Self::EitherAccess => None,
        }
    }
}

/// State per location per node keeping track of where relative to this
/// node exposed nodes are and what access permissions they have.
///
/// Designed to be completely determined by its parent, siblings and
/// direct children's max_local_access/max_foreign_access.
#[derive(Clone, Default, PartialEq, Eq)]
pub struct WildcardState {
    /// How many of this node's direct children have `max_local_access()==Write`.
    child_writes: u16,
    /// How many of this node's direct children have `max_local_access()>=Read`.
    child_reads: u16,
    /// The maximum access level that could happen from an exposed node
    /// that is foreign to this node.
    ///
    /// This is calculated as the `max()` of the parent's `max_foreign_access`,
    /// `exposed_as` and the siblings' `max_local_access()`.
    max_foreign_access: WildcardAccessLevel,
    /// At what access level this node itself is exposed.
    exposed_as: WildcardAccessLevel,
}
impl Debug for WildcardState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WildcardState")
            .field("child_r/w", &(self.child_reads, self.child_writes))
            .field("foreign", &self.max_foreign_access)
            .field("exposed_as", &self.exposed_as)
            .finish()
    }
}
impl WildcardState {
    /// The maximum access level that could happen from an exposed
    /// node that is local to this node.
    fn max_local_access(&self) -> WildcardAccessLevel {
        use WildcardAccessLevel::*;
        max(
            self.exposed_as,
            if self.child_writes > 0 {
                Write
            } else if self.child_reads > 0 {
                Read
            } else {
                None
            },
        )
    }

    /// From where relative to the node with this wildcard info a read or write access could happen.
    /// If `only_foreign` is true then we treat `LocalAccess` as impossible. This means we return
    /// `None` if only a `LocalAccess` is possible, and we treat `EitherAccess` as a
    /// `ForeignAccess`.
    pub fn access_relatedness(
        &self,
        kind: AccessKind,
        only_foreign: bool,
    ) -> Option<WildcardAccessRelatedness> {
        let rel = match kind {
            AccessKind::Read => self.read_access_relatedness(),
            AccessKind::Write => self.write_access_relatedness(),
        };
        if only_foreign {
            use WildcardAccessRelatedness as E;
            match rel {
                Some(E::EitherAccess | E::ForeignAccess) => Some(E::ForeignAccess),
                Some(E::LocalAccess) | None => None,
            }
        } else {
            rel
        }
    }

    /// From where relative to the node with this wildcard info a read access could happen.
    fn read_access_relatedness(&self) -> Option<WildcardAccessRelatedness> {
        let has_foreign = self.max_foreign_access >= WildcardAccessLevel::Read;
        let has_local = self.max_local_access() >= WildcardAccessLevel::Read;
        use WildcardAccessRelatedness as E;
        match (has_foreign, has_local) {
            (true, true) => Some(E::EitherAccess),
            (true, false) => Some(E::ForeignAccess),
            (false, true) => Some(E::LocalAccess),
            (false, false) => None,
        }
    }

    /// From where relative to the node with this wildcard info a write access could happen.
    fn write_access_relatedness(&self) -> Option<WildcardAccessRelatedness> {
        let has_foreign = self.max_foreign_access == WildcardAccessLevel::Write;
        let has_local = self.max_local_access() == WildcardAccessLevel::Write;
        use WildcardAccessRelatedness as E;
        match (has_foreign, has_local) {
            (true, true) => Some(E::EitherAccess),
            (true, false) => Some(E::ForeignAccess),
            (false, true) => Some(E::LocalAccess),
            (false, false) => None,
        }
    }

    /// Gets the access tracking information for a new child node of a parent with this
    /// wildcard info.
    /// The new node doesn't have any child reads/writes, but calculates `max_foreign_access`
    /// from its parent.
    pub fn for_new_child(&self) -> Self {
        Self {
            max_foreign_access: max(self.max_foreign_access, self.max_local_access()),
            ..Default::default()
        }
    }
    /// Crates the initial `WildcardState` for a wildcard root.
    /// This has `max_foreign_access==Write` as it actually is the child of *some* exposed node
    /// through which we can receive foreign accesses.
    ///
    /// This is different from the main root which has `max_foreign_access==None`, since there
    /// cannot be a foreign access to the root of the allocation.
    pub fn for_wildcard_root() -> Self {
        Self { max_foreign_access: WildcardAccessLevel::Write, ..Default::default() }
    }

    /// Pushes the nodes of `children` onto the stack who's `max_foreign_access`
    /// needs to be updated.
    ///
    /// * `children`: A list of nodes with the same parent. `children` doesn't
    ///   necessarily have to contain all children of parent, but can just be
    ///   a subset.
    ///
    /// * `child_reads`, `child_writes`: How many of `children` have `max_local_access()`
    ///   of at least `read`/`write`
    ///
    /// * `new_foreign_access`, `old_foreign_access`:
    ///   The max possible access level that is foreign to all `children`
    ///   (i.e., it is not local to *any* of them).
    ///   This can be calculated as the max of the parent's `exposed_as()`, `max_foreign_access`
    ///   and of all `max_local_access()` of any nodes with the same parent that are
    ///   not listed in `children`.
    ///
    ///   This access level changed from `old` to `new`, which is why we need to
    ///   update `children`.
    fn push_relevant_children(
        stack: &mut Vec<(UniIndex, WildcardAccessLevel)>,
        new_foreign_access: WildcardAccessLevel,
        old_foreign_access: WildcardAccessLevel,
        child_reads: u16,
        child_writes: u16,
        children: impl Iterator<Item = UniIndex>,

        wildcard_accesses: &UniValMap<WildcardState>,
    ) {
        use WildcardAccessLevel::*;

        // Nothing changed so we don't need to update anything.
        if new_foreign_access == old_foreign_access {
            return;
        }

        // We need to consider that the children's `max_local_access()` affect each
        // other's `max_foreign_access`, but do not affect their own `max_foreign_access`.

        // The new `max_foreign_acces` for children with `max_local_access()==Write`.
        let write_foreign_access = max(
            new_foreign_access,
            if child_writes > 1 {
                // There exists at least one more child with exposed write access.
                // This means that a foreign write through that node is possible.
                Write
            } else if child_reads > 1 {
                // There exists at least one more child with exposed read access,
                // but no other with write access.
                // This means that a foreign read but no write through that node
                // is possible.
                Read
            } else {
                // There are no other nodes with read or write access.
                // This means no foreign writes through other children are possible.
                None
            },
        );

        // The new `max_foreign_acces` for children with `max_local_access()==Read`.
        let read_foreign_access = max(
            new_foreign_access,
            if child_writes > 0 {
                // There exists at least one child with write access (and it's not this one).
                Write
            } else if child_reads > 1 {
                // There exists at least one more child with exposed read access,
                // but no other with write access.
                Read
            } else {
                // There are no other nodes with read or write access,
                None
            },
        );

        // The new `max_foreign_acces` for children with `max_local_access()==None`.
        let none_foreign_access = max(
            new_foreign_access,
            if child_writes > 0 {
                // There exists at least one child with write access (and it's not this one).
                Write
            } else if child_reads > 0 {
                // There exists at least one child with read access (and it's not this one),
                // but none with write access.
                Read
            } else {
                // No children are exposed as read or write.
                None
            },
        );

        stack.extend(children.filter_map(|child| {
            let state = wildcard_accesses.get(child).cloned().unwrap_or_default();

            let new_foreign_access = match state.max_local_access() {
                Write => write_foreign_access,
                Read => read_foreign_access,
                None => none_foreign_access,
            };

            if new_foreign_access != state.max_foreign_access {
                Some((child, new_foreign_access))
            } else {
                Option::None
            }
        }));
    }

    /// Update the tracking information of a tree, to reflect that the node specified by `id` is
    /// now exposed with `new_exposed_as`.
    ///
    /// Propagates the Willard access information over the tree. This needs to be called every
    /// time the access level of an exposed node changes, to keep the state in sync with
    /// the rest of the tree.
    pub fn update_exposure(
        id: UniIndex,
        new_exposed_as: WildcardAccessLevel,
        nodes: &UniValMap<Node>,
        wildcard_accesses: &mut UniValMap<WildcardState>,
    ) {
        let mut entry = wildcard_accesses.entry(id);
        let src_state = entry.or_insert(Default::default());
        let old_exposed_as = src_state.exposed_as;

        // If the exposure doesn't change, then we don't need to update anything.
        if old_exposed_as == new_exposed_as {
            return;
        }

        let src_old_local_access = src_state.max_local_access();

        src_state.exposed_as = new_exposed_as;

        let src_new_local_access = src_state.max_local_access();

        // Stack of nodes for which the max_foreign_access field needs to be updated.
        // Will be filled with the children of this node and its parents children before
        // we begin downwards traversal.
        let mut stack: Vec<(UniIndex, WildcardAccessLevel)> = Vec::new();

        // Add the direct children of this node to the stack.
        {
            let node = nodes.get(id).unwrap();
            Self::push_relevant_children(
                &mut stack,
                // new_foreign_access
                max(src_state.max_foreign_access, new_exposed_as),
                // old_foreign_access
                max(src_state.max_foreign_access, old_exposed_as),
                // Consider all children.
                src_state.child_reads,
                src_state.child_writes,
                node.children.iter().copied(),
                wildcard_accesses,
            );
        }
        // We need to propagate the tracking info up the tree, for this we traverse
        // up the parents.
        // We can skip propagating info to the parent and siblings of a node if its
        // access didn't change.
        {
            // The child from which we came.
            let mut child = id;
            // This is the `max_local_access()` of the child we came from, before
            // this update...
            let mut old_child_access = src_old_local_access;
            // and after this update.
            let mut new_child_access = src_new_local_access;
            while let Some(parent_id) = nodes.get(child).unwrap().parent {
                let parent_node = nodes.get(parent_id).unwrap();
                let mut entry = wildcard_accesses.entry(parent_id);
                let parent_state = entry.or_insert(Default::default());

                let old_parent_local_access = parent_state.max_local_access();
                use WildcardAccessLevel::*;
                // Updating this node's tracking state for its children.
                match (old_child_access, new_child_access) {
                    (None | Read, Write) => parent_state.child_writes += 1,
                    (Write, None | Read) => parent_state.child_writes -= 1,
                    _ => {}
                }
                match (old_child_access, new_child_access) {
                    (None, Read | Write) => parent_state.child_reads += 1,
                    (Read | Write, None) => parent_state.child_reads -= 1,
                    _ => {}
                }

                let new_parent_local_access = parent_state.max_local_access();

                {
                    // We need to update the `max_foreign_access` of `child`'s
                    // siblings. For this we can reuse the `push_relevant_children`
                    // function.
                    //
                    // We pass it just the siblings without child itself. Since
                    // `child`'s `max_local_access()` is foreign to all of its
                    // siblings we can pass it as part of the foreign access.

                    let parent_access =
                        max(parent_state.exposed_as, parent_state.max_foreign_access);
                    // This is how many of `child`'s siblings have read/write local access.
                    // If `child` itself has access, then we need to subtract its access from the count.
                    let sibling_reads =
                        parent_state.child_reads - if new_child_access >= Read { 1 } else { 0 };
                    let sibling_writes =
                        parent_state.child_writes - if new_child_access >= Write { 1 } else { 0 };
                    Self::push_relevant_children(
                        &mut stack,
                        // new_foreign_access
                        max(parent_access, new_child_access),
                        // old_foreign_access
                        max(parent_access, old_child_access),
                        // Consider only siblings of child.
                        sibling_reads,
                        sibling_writes,
                        parent_node.children.iter().copied().filter(|id| child != *id),
                        wildcard_accesses,
                    );
                }
                if old_parent_local_access == new_parent_local_access {
                    // We didn't change `max_local_access()` for parent, so we don't need to propagate further upwards.
                    break;
                }

                old_child_access = old_parent_local_access;
                new_child_access = new_parent_local_access;
                child = parent_id;
            }
        }
        // Traverses down the tree to update max_foreign_access fields of children and cousins who need to be updated.
        while let Some((id, new_access)) = stack.pop() {
            let node = nodes.get(id).unwrap();
            let mut entry = wildcard_accesses.entry(id);
            let state = entry.or_insert(Default::default());

            let old_access = state.max_foreign_access;
            state.max_foreign_access = new_access;

            Self::push_relevant_children(
                &mut stack,
                // new_foreign_access
                max(state.exposed_as, new_access),
                // old_foreign_access
                max(state.exposed_as, old_access),
                // Consider all children.
                state.child_reads,
                state.child_writes,
                node.children.iter().copied(),
                wildcard_accesses,
            );
        }
    }
}

impl Tree {
    /// Marks the tag as exposed & updates the wildcard tracking data structure
    /// to represent its access level.
    /// Also takes as an argument whether the tag is protected or not.
    pub fn expose_tag(&mut self, tag: BorTag, protected: bool) {
        let id = self.tag_mapping.get(&tag).unwrap();
        let node = self.nodes.get_mut(id).unwrap();
        node.is_exposed = true;
        let node = self.nodes.get(id).unwrap();

        // When the first tag gets exposed then we initialize the
        // wildcard state for every node and location in the tree.
        for (_, loc) in self.locations.iter_mut_all() {
            let perm = loc
                .perms
                .get(id)
                .map(|p| p.permission())
                .unwrap_or_else(|| node.default_location_state().permission());

            let access_type = perm.strongest_allowed_local_access(protected);
            WildcardState::update_exposure(
                id,
                access_type,
                &self.nodes,
                &mut loc.wildcard_accesses,
            );
        }
    }

    /// This updates the wildcard tracking data structure to reflect the release of
    /// the protector on `tag`.
    pub(super) fn update_exposure_for_protector_release(&mut self, tag: BorTag) {
        let idx = self.tag_mapping.get(&tag).unwrap();

        // We check if the node is already exposed, as we don't want to expose any
        // nodes which aren't already exposed.

        if self.nodes.get(idx).unwrap().is_exposed {
            // Updates the exposure to the new permission on every location.
            self.expose_tag(tag, /* protected */ false);
        }
    }
}

#[cfg(feature = "expensive-consistency-checks")]
impl Tree {
    /// Checks that the wildcard tracking data structure is internally consistent and
    /// has the correct `exposed_as` values.
    pub fn verify_wildcard_consistency(&self, global: &GlobalState) {
        // We rely on the fact that `roots` is ordered according to tag from low to high.
        assert!(self.roots.is_sorted_by_key(|idx| self.nodes.get(*idx).unwrap().tag));
        let main_root_idx = self.roots[0];

        let protected_tags = &global.borrow().protected_tags;
        for (_, loc) in self.locations.iter_all() {
            let wildcard_accesses = &loc.wildcard_accesses;
            let perms = &loc.perms;
            // Checks if accesses is empty.
            if wildcard_accesses.is_empty() {
                return;
            }
            for (id, node) in self.nodes.iter() {
                let state = wildcard_accesses.get(id).unwrap();

                let expected_exposed_as = if node.is_exposed {
                    let perm =
                        perms.get(id).copied().unwrap_or_else(|| node.default_location_state());

                    perm.permission()
                        .strongest_allowed_local_access(protected_tags.contains_key(&node.tag))
                } else {
                    WildcardAccessLevel::None
                };

                // The foreign wildcard accesses possible at a node are determined by which
                // accesses can originate from their siblings, their parent, and from above
                // their parent.
                let expected_max_foreign_access = if let Some(parent) = node.parent {
                    let parent_node = self.nodes.get(parent).unwrap();
                    let parent_state = wildcard_accesses.get(parent).unwrap();

                    let max_sibling_access = parent_node
                        .children
                        .iter()
                        .copied()
                        .filter(|child| *child != id)
                        .map(|child| {
                            let state = wildcard_accesses.get(child).unwrap();
                            state.max_local_access()
                        })
                        .fold(WildcardAccessLevel::None, max);

                    max_sibling_access
                        .max(parent_state.max_foreign_access)
                        .max(parent_state.exposed_as)
                } else {
                    if main_root_idx == id {
                        // There can never be a foreign access to the root of the allocation.
                        // So its foreign access level is always `None`.
                        WildcardAccessLevel::None
                    } else {
                        // For wildcard roots any access on a different subtree can be foreign
                        // to it. So a wildcard root has the maximum possible foreign access
                        // level.
                        WildcardAccessLevel::Write
                    }
                };

                // Count how many children can be the source of wildcard reads or writes
                // (either directly, or via their children).
                let child_accesses = node.children.iter().copied().map(|child| {
                    let state = wildcard_accesses.get(child).unwrap();
                    state.max_local_access()
                });
                let expected_child_reads =
                    child_accesses.clone().filter(|a| *a >= WildcardAccessLevel::Read).count();
                let expected_child_writes =
                    child_accesses.filter(|a| *a >= WildcardAccessLevel::Write).count();

                assert_eq!(
                    expected_exposed_as, state.exposed_as,
                    "tag {:?} (id:{id:?}) should be exposed as {expected_exposed_as:?} but is exposed as {:?}",
                    node.tag, state.exposed_as
                );
                assert_eq!(
                    expected_max_foreign_access, state.max_foreign_access,
                    "expected {:?}'s (id:{id:?}) max_foreign_access to be {:?} instead of {:?}",
                    node.tag, expected_max_foreign_access, state.max_foreign_access
                );
                let child_reads: usize = state.child_reads.into();
                assert_eq!(
                    expected_child_reads, child_reads,
                    "expected {:?}'s (id:{id:?}) child_reads to be {} instead of {}",
                    node.tag, expected_child_reads, child_reads
                );
                let child_writes: usize = state.child_writes.into();
                assert_eq!(
                    expected_child_writes, child_writes,
                    "expected {:?}'s (id:{id:?}) child_writes to be {} instead of {}",
                    node.tag, expected_child_writes, child_writes
                );
            }
        }
    }
}
