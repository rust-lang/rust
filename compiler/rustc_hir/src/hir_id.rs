use crate::def_id::{DefId, DefIndex, LocalDefId, CRATE_DEF_ID};
use rustc_data_structures::stable_hasher::{HashStable, StableHasher, StableOrd, ToStableHashKey};
use rustc_span::{def_id::DefPathHash, HashStableContext};
use std::fmt::{self, Debug};

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
#[derive(Encodable, Decodable)]
pub struct OwnerId {
    pub def_id: LocalDefId,
}

impl Debug for OwnerId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Example: DefId(0:1 ~ aa[7697]::{use#0})
        Debug::fmt(&self.def_id, f)
    }
}

impl From<OwnerId> for HirId {
    fn from(owner: OwnerId) -> HirId {
        HirId { owner, local_id: ItemLocalId::from_u32(0) }
    }
}

impl From<OwnerId> for DefId {
    fn from(value: OwnerId) -> Self {
        value.to_def_id()
    }
}

impl OwnerId {
    #[inline]
    pub fn to_def_id(self) -> DefId {
        self.def_id.to_def_id()
    }
}

impl rustc_index::Idx for OwnerId {
    #[inline]
    fn new(idx: usize) -> Self {
        OwnerId { def_id: LocalDefId { local_def_index: DefIndex::from_usize(idx) } }
    }

    #[inline]
    fn index(self) -> usize {
        self.def_id.local_def_index.as_usize()
    }
}

impl<CTX: HashStableContext> HashStable<CTX> for OwnerId {
    #[inline]
    fn hash_stable(&self, hcx: &mut CTX, hasher: &mut StableHasher) {
        self.to_stable_hash_key(hcx).hash_stable(hcx, hasher);
    }
}

impl<CTX: HashStableContext> ToStableHashKey<CTX> for OwnerId {
    type KeyType = DefPathHash;

    #[inline]
    fn to_stable_hash_key(&self, hcx: &CTX) -> DefPathHash {
        hcx.def_path_hash(self.to_def_id())
    }
}

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
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
#[derive(Encodable, Decodable, HashStable_Generic)]
#[rustc_pass_by_value]
pub struct HirId {
    pub owner: OwnerId,
    pub local_id: ItemLocalId,
}

impl Debug for HirId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Example: HirId(DefId(0:1 ~ aa[7697]::{use#0}).10)
        // Don't use debug_tuple to always keep this on one line.
        write!(f, "HirId({:?}.{:?})", self.owner, self.local_id)
    }
}

impl HirId {
    /// Signal local id which should never be used.
    pub const INVALID: HirId =
        HirId { owner: OwnerId { def_id: CRATE_DEF_ID }, local_id: ItemLocalId::INVALID };

    #[inline]
    pub fn expect_owner(self) -> OwnerId {
        assert_eq!(self.local_id.index(), 0);
        self.owner
    }

    #[inline]
    pub fn as_owner(self) -> Option<OwnerId> {
        if self.local_id.index() == 0 { Some(self.owner) } else { None }
    }

    #[inline]
    pub fn is_owner(self) -> bool {
        self.local_id.index() == 0
    }

    #[inline]
    pub fn make_owner(owner: LocalDefId) -> Self {
        Self { owner: OwnerId { def_id: owner }, local_id: ItemLocalId::from_u32(0) }
    }

    pub fn index(self) -> (usize, usize) {
        (rustc_index::Idx::index(self.owner.def_id), rustc_index::Idx::index(self.local_id))
    }
}

impl fmt::Display for HirId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl Ord for HirId {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (self.index()).cmp(&(other.index()))
    }
}

impl PartialOrd for HirId {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

rustc_data_structures::define_stable_id_collections!(HirIdMap, HirIdSet, HirIdMapEntry, HirId);
rustc_data_structures::define_id_collections!(
    ItemLocalMap,
    ItemLocalSet,
    ItemLocalMapEntry,
    ItemLocalId
);

rustc_index::newtype_index! {
    /// An `ItemLocalId` uniquely identifies something within a given "item-like";
    /// that is, within a `hir::Item`, `hir::TraitItem`, or `hir::ImplItem`. There is no
    /// guarantee that the numerical value of a given `ItemLocalId` corresponds to
    /// the node's position within the owning item in any way, but there is a
    /// guarantee that the `ItemLocalId`s within an owner occupy a dense range of
    /// integers starting at zero, so a mapping that maps all or most nodes within
    /// an "item-like" to something else can be implemented by a `Vec` instead of a
    /// tree or hash map.
    #[derive(HashStable_Generic)]
    pub struct ItemLocalId {}
}

impl ItemLocalId {
    /// Signal local id which should never be used.
    pub const INVALID: ItemLocalId = ItemLocalId::MAX;
}

// Safety: Ord is implement as just comparing the ItemLocalId's numerical
// values and these are not changed by (de-)serialization.
unsafe impl StableOrd for ItemLocalId {
    const CAN_USE_UNSTABLE_SORT: bool = true;
}

/// The `HirId` corresponding to `CRATE_NODE_ID` and `CRATE_DEF_ID`.
pub const CRATE_HIR_ID: HirId =
    HirId { owner: OwnerId { def_id: CRATE_DEF_ID }, local_id: ItemLocalId::from_u32(0) };

pub const CRATE_OWNER_ID: OwnerId = OwnerId { def_id: CRATE_DEF_ID };
