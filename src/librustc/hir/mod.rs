//! HIR datatypes. See the [rustc dev guide] for more info.
//!
//! [rustc dev guide]: https://rustc-dev-guide.rust-lang.org/hir.html

pub mod exports;
pub mod map;

use crate::ty::query::Providers;
use crate::ty::TyCtxt;
use rustc_data_structures::cold_path;
use rustc_data_structures::fx::FxHashMap;
use rustc_hir::def_id::{DefId, LOCAL_CRATE};
use rustc_hir::print;
use rustc_hir::Body;
use rustc_hir::Crate;
use rustc_hir::HirId;
use rustc_hir::ItemLocalId;
use rustc_hir::Node;
use rustc_index::vec::IndexVec;
use std::ops::Deref;

#[derive(HashStable)]
pub struct HirOwner<'tcx> {
    parent: HirId,
    node: Node<'tcx>,
}

#[derive(HashStable, Clone)]
pub struct HirItem<'tcx> {
    parent: ItemLocalId,
    node: Node<'tcx>,
}

#[derive(HashStable)]
pub struct HirOwnerItems<'tcx> {
    //owner: &'tcx HirOwner<'tcx>,
    items: IndexVec<ItemLocalId, Option<HirItem<'tcx>>>,
    bodies: FxHashMap<ItemLocalId, &'tcx Body<'tcx>>,
}

/// A wrapper type which allows you to access HIR.
#[derive(Clone)]
pub struct Hir<'tcx> {
    tcx: TyCtxt<'tcx>,
    map: &'tcx map::Map<'tcx>,
}

impl<'tcx> Hir<'tcx> {
    pub fn krate(&self) -> &'tcx Crate<'tcx> {
        self.tcx.hir_crate(LOCAL_CRATE)
    }
}

impl<'tcx> Deref for Hir<'tcx> {
    type Target = &'tcx map::Map<'tcx>;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.map
    }
}

impl<'hir> print::PpAnn for Hir<'hir> {
    fn nested(&self, state: &mut print::State<'_>, nested: print::Nested) {
        self.map.nested(state, nested)
    }
}

impl<'tcx> TyCtxt<'tcx> {
    #[inline(always)]
    pub fn hir(self) -> Hir<'tcx> {
        let map = self.late_hir_map.load();
        let map = if unlikely!(map.is_none()) {
            cold_path(|| {
                let map = self.hir_map(LOCAL_CRATE);
                self.late_hir_map.store(Some(map));
                map
            })
        } else {
            map.unwrap()
        };
        Hir { tcx: self, map }
    }

    pub fn parent_module(self, id: HirId) -> DefId {
        self.parent_module_from_def_id(DefId::local(id.owner))
    }
}

pub fn provide(providers: &mut Providers<'_>) {
    providers.parent_module_from_def_id = |tcx, id| {
        let hir = tcx.hir();
        hir.local_def_id(hir.get_module_parent_node(hir.as_local_hir_id(id).unwrap()))
    };
    providers.hir_crate = |tcx, _| tcx.hir_map(LOCAL_CRATE).untracked_krate();
    providers.hir_map = |tcx, id| {
        assert_eq!(id, LOCAL_CRATE);
        let early = tcx.hir_map.steal();
        tcx.arena.alloc(map::Map {
            tcx,
            krate: early.krate,

            dep_graph: early.dep_graph,

            crate_hash: early.crate_hash,

            owner_map: early.owner_map,
            owner_items_map: early.owner_items_map,

            definitions: early.definitions,

            hir_to_node_id: early.hir_to_node_id,
        })
    };
    providers.hir_module_items = |tcx, id| {
        assert_eq!(id.krate, LOCAL_CRATE);
        let hir = tcx.hir();
        let module = hir.as_local_hir_id(id).unwrap();
        &hir.untracked_krate().modules[&module]
    };
    providers.hir_owner = |tcx, id| {
        assert_eq!(id.krate, LOCAL_CRATE);
        *tcx.hir().map.owner_map.get(&id.index).unwrap()
    };
    providers.hir_owner_items = |tcx, id| {
        assert_eq!(id.krate, LOCAL_CRATE);
        *tcx.hir().map.owner_items_map.get(&id.index).unwrap()
    };
    map::provide(providers);
}
