use crate::def_id::{LocalDefId, CRATE_DEF_INDEX};
use rustc_index::vec::IndexVec;
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
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
#[derive(Encodable, Decodable)]
pub struct HirId {
    pub owner: LocalDefId,
    pub local_id: ItemLocalId,
}

impl HirId {
    pub fn expect_owner(self) -> LocalDefId {
        assert_eq!(self.local_id.index(), 0);
        self.owner
    }

    pub fn as_owner(self) -> Option<LocalDefId> {
        if self.local_id.index() == 0 { Some(self.owner) } else { None }
    }

    #[inline]
    pub fn make_owner(owner: LocalDefId) -> Self {
        Self { owner, local_id: ItemLocalId::from_u32(0) }
    }
}

impl fmt::Display for HirId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
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

/// N.B. This collection is currently unused, but will be used by #72015 and future PRs.
#[derive(Clone, Default, Debug, Encodable, Decodable)]
pub struct HirIdVec<T> {
    map: IndexVec<LocalDefId, IndexVec<ItemLocalId, T>>,
}

impl<T> HirIdVec<T> {
    pub fn push_owner(&mut self, id: LocalDefId) {
        self.map.ensure_contains_elem(id, IndexVec::new);
    }

    pub fn push(&mut self, id: HirId, value: T) {
        if id.local_id == ItemLocalId::from_u32(0) {
            self.push_owner(id.owner);
        }
        let submap = &mut self.map[id.owner];
        let _ret_id = submap.push(value);
        debug_assert_eq!(_ret_id, id.local_id);
    }

    pub fn push_sparse(&mut self, id: HirId, value: T)
    where
        T: Default,
    {
        self.map.ensure_contains_elem(id.owner, IndexVec::new);
        let submap = &mut self.map[id.owner];
        let i = id.local_id.index();
        let len = submap.len();
        if i >= len {
            submap.extend(std::iter::repeat_with(T::default).take(i - len + 1));
        }
        submap[id.local_id] = value;
    }

    pub fn get(&self, id: HirId) -> Option<&T> {
        self.map.get(id.owner)?.get(id.local_id)
    }

    pub fn get_owner(&self, id: LocalDefId) -> &IndexVec<ItemLocalId, T> {
        &self.map[id]
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.map.iter().flat_map(|la| la.iter())
    }

    pub fn iter_enumerated(&self) -> impl Iterator<Item = (HirId, &T)> {
        self.map.iter_enumerated().flat_map(|(owner, la)| {
            la.iter_enumerated().map(move |(local_id, attr)| (HirId { owner, local_id }, attr))
        })
    }
}

impl<T> std::ops::Index<HirId> for HirIdVec<T> {
    type Output = T;

    fn index(&self, id: HirId) -> &T {
        &self.map[id.owner][id.local_id]
    }
}

impl<T> std::ops::IndexMut<HirId> for HirIdVec<T> {
    fn index_mut(&mut self, id: HirId) -> &mut T {
        &mut self.map[id.owner][id.local_id]
    }
}
