//! HIR datatypes. See the [rustc dev guide] for more info.
//!
//! [rustc dev guide]: https://rustc-dev-guide.rust-lang.org/hir.html

pub mod exports;
pub mod map;
pub mod place;

use crate::ich::StableHashingContext;
use crate::ty::query::Providers;
use crate::ty::TyCtxt;
use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_hir::def_id::{LocalDefId, LOCAL_CRATE};
use rustc_hir::*;
use rustc_index::vec::IndexVec;

pub struct Owner<'tcx> {
    parent: HirId,
    node: Node<'tcx>,
}

impl<'a, 'tcx> HashStable<StableHashingContext<'a>> for Owner<'tcx> {
    fn hash_stable(&self, hcx: &mut StableHashingContext<'a>, hasher: &mut StableHasher) {
        let Owner { parent, node } = self;
        hcx.while_hashing_hir_bodies(false, |hcx| {
            parent.hash_stable(hcx, hasher);
            node.hash_stable(hcx, hasher);
        });
    }
}

#[derive(Clone)]
pub struct ParentedNode<'tcx> {
    parent: ItemLocalId,
    node: Node<'tcx>,
}

pub struct OwnerNodes<'tcx> {
    hash: Fingerprint,
    nodes: IndexVec<ItemLocalId, Option<ParentedNode<'tcx>>>,
    bodies: FxHashMap<ItemLocalId, &'tcx Body<'tcx>>,
}

impl<'a, 'tcx> HashStable<StableHashingContext<'a>> for OwnerNodes<'tcx> {
    fn hash_stable(&self, hcx: &mut StableHashingContext<'a>, hasher: &mut StableHasher) {
        // We ignore the `nodes` and `bodies` fields since these refer to information included in
        // `hash` which is hashed in the collector and used for the crate hash.
        let OwnerNodes { hash, nodes: _, bodies: _ } = *self;
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

pub fn provide(providers: &mut Providers) {
    providers.parent_module_from_def_id = |tcx, id| {
        let hir = tcx.hir();
        hir.local_def_id(hir.get_module_parent_node(hir.local_def_id_to_hir_id(id)))
    };
    providers.hir_crate = |tcx, _| tcx.untracked_crate;
    providers.index_hir = map::index_hir;
    providers.hir_module_items = |tcx, id| {
        let hir = tcx.hir();
        let module = hir.local_def_id_to_hir_id(id);
        &tcx.untracked_crate.modules[&module]
    };
    providers.hir_owner = |tcx, id| tcx.index_hir(LOCAL_CRATE).map[id].signature;
    providers.hir_owner_nodes = |tcx, id| tcx.index_hir(LOCAL_CRATE).map[id].with_bodies.as_deref();
    providers.fn_arg_names = |tcx, id| {
        let hir = tcx.hir();
        let hir_id = hir.local_def_id_to_hir_id(id.expect_local());
        if let Some(body_id) = hir.maybe_body_owned_by(hir_id) {
            tcx.arena.alloc_from_iter(hir.body_param_names(body_id))
        } else if let Node::TraitItem(&TraitItem {
            kind: TraitItemKind::Fn(_, TraitFn::Required(idents)),
            ..
        }) = hir.get(hir_id)
        {
            tcx.arena.alloc_slice(idents)
        } else {
            span_bug!(hir.span(hir_id), "fn_arg_names: unexpected item {:?}", id);
        }
    };
    map::provide(providers);
}
