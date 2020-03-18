//! HIR datatypes. See the [rustc dev guide] for more info.
//!
//! [rustc dev guide]: https://rustc-dev-guide.rust-lang.org/hir.html

pub mod exports;
pub mod map;

use crate::ich::StableHashingContext;
use crate::ty::query::Providers;
use crate::ty::TyCtxt;
use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_hir::def_id::{LocalDefId, LOCAL_CRATE};
use rustc_hir::Body;
use rustc_hir::HirId;
use rustc_hir::ItemLocalId;
use rustc_hir::Node;
use rustc_index::vec::IndexVec;

pub struct HirOwner<'tcx> {
    parent: HirId,
    node: Node<'tcx>,
}

impl<'a, 'tcx> HashStable<StableHashingContext<'a>> for HirOwner<'tcx> {
    fn hash_stable(&self, hcx: &mut StableHashingContext<'a>, hasher: &mut StableHasher) {
        let HirOwner { parent, node } = self;
        hcx.while_hashing_hir_bodies(false, |hcx| {
            parent.hash_stable(hcx, hasher);
            node.hash_stable(hcx, hasher);
        });
    }
}

#[derive(Clone)]
pub struct HirItem<'tcx> {
    parent: ItemLocalId,
    node: Node<'tcx>,
}

pub struct HirOwnerItems<'tcx> {
    hash: Fingerprint,
    items: IndexVec<ItemLocalId, Option<HirItem<'tcx>>>,
    bodies: FxHashMap<ItemLocalId, &'tcx Body<'tcx>>,
}

impl<'a, 'tcx> HashStable<StableHashingContext<'a>> for HirOwnerItems<'tcx> {
    fn hash_stable(&self, hcx: &mut StableHashingContext<'a>, hasher: &mut StableHasher) {
        // We ignore the `items` and `bodies` fields since these refer to information included in
        // `hash` which is hashed in the collector and used for the crate hash.
        let HirOwnerItems { hash, items: _, bodies: _ } = *self;
        hash.hash_stable(hcx, hasher);
    }
}

impl<'tcx> TyCtxt<'tcx> {
    #[inline(always)]
    pub fn hir(self) -> map::Map<'tcx> {
        map::Map { tcx: self }
    }

    pub fn parent_module(self, id: HirId) -> LocalDefId {
        self.parent_module_from_def_id(id.owner)
    }
}

pub fn provide(providers: &mut Providers<'_>) {
    providers.parent_module_from_def_id = |tcx, id| {
        let hir = tcx.hir();
        hir.local_def_id(hir.get_module_parent_node(hir.as_local_hir_id(id.to_def_id()).unwrap()))
            .expect_local()
    };
    providers.hir_crate = |tcx, _| tcx.untracked_crate;
    providers.index_hir = map::index_hir;
    providers.hir_module_items = |tcx, id| {
        let hir = tcx.hir();
        let module = hir.as_local_hir_id(id.to_def_id()).unwrap();
        &tcx.untracked_crate.modules[&module]
    };
    providers.hir_owner = |tcx, id| tcx.index_hir(LOCAL_CRATE).map[id].signature.unwrap();
    providers.hir_owner_items = |tcx, id| {
        tcx.index_hir(LOCAL_CRATE).map[id].with_bodies.as_ref().map(|items| &**items).unwrap()
    };
    map::provide(providers);
}
