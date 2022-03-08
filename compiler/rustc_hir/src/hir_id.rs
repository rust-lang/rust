use crate::def_id::{LocalDefId, CRATE_DEF_INDEX};
use std::fmt;

/// Uniquely identifies a node in the HIR of the current crate. It is
/// composed of the `owner`, which is the `LocalDefId` of the directly enclosing
/// `hir::Item`, `hir::TraitItem`, or `hir::ImplItem` (i.e., the closest "item-like"),
/// and the `local_id` which is unique within the given owner.
///
/// This two-level structure makes for more stable values: One can move an item
/// around within the source code, or add or remove stuff before it, without
/// the `local_id` part of the `HirId` changing, which is a very useful property in
/// incremental compilation where we have to persist things through changes to
/// the code base.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
#[derive(Encodable, Decodable)]
pub struct HirId {
    pub owner: LocalDefId,
    pub local_id: ItemLocalId,
}

impl HirId {
    #[inline]
    pub fn expect_owner(self) -> LocalDefId {
        assert_eq!(self.local_id.index(), 0);
        self.owner
    }

    #[inline]
    pub fn as_owner(self) -> Option<LocalDefId> {
        if self.local_id.index() == 0 { Some(self.owner) } else { None }
    }

    #[inline]
    pub fn is_owner(self) -> bool {
        self.local_id.index() == 0
    }

    #[inline]
    pub fn make_owner(owner: LocalDefId) -> Self {
        Self { owner, local_id: ItemLocalId::from_u32(0) }
    }

    pub fn index(self) -> (usize, usize) {
        (rustc_index::vec::Idx::index(self.owner), rustc_index::vec::Idx::index(self.local_id))
    }
}

impl fmt::Display for HirId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl Ord for HirId {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (self.index()).cmp(&(other.index()))
    }
}

impl PartialOrd for HirId {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(&other))
    }
}

rustc_data_structures::define_id_collections!(HirIdMap, HirIdSet, HirId);
rustc_data_structures::define_id_collections!(ItemLocalMap, ItemLocalSet, ItemLocalId);

rustc_index::newtype_index! {
    /// An `ItemLocalId` uniquely identifies something within a given "item-like";
    /// that is, within a `hir::Item`, `hir::TraitItem`, or `hir::ImplItem`. There is no
    /// guarantee that the numerical value of a given `ItemLocalId` corresponds to
    /// the node's position within the owning item in any way, but there is a
    /// guarantee that the `LocalItemId`s within an owner occupy a dense range of
    /// integers starting at zero, so a mapping that maps all or most nodes within
    /// an "item-like" to something else can be implemented by a `Vec` instead of a
    /// tree or hash map.
    pub struct ItemLocalId { .. }
}
rustc_data_structures::impl_stable_hash_via_hash!(ItemLocalId);
impl ItemLocalId {
    /// Signal local id which should never be used.
    pub const INVALID: ItemLocalId = ItemLocalId::MAX;
}

/// The `HirId` corresponding to `CRATE_NODE_ID` and `CRATE_DEF_INDEX`.
pub const CRATE_HIR_ID: HirId = HirId {
    owner: LocalDefId { local_def_index: CRATE_DEF_INDEX },
    local_id: ItemLocalId::from_u32(0),
};
