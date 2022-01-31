//! HIR datatypes. See the [rustc dev guide] for more info.
//!
//! [rustc dev guide]: https://rustc-dev-guide.rust-lang.org/hir.html

pub mod map;
pub mod nested_filter;
pub mod place;

use crate::ty::query::Providers;
use crate::ty::TyCtxt;
use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_hir::def_id::LocalDefId;
use rustc_hir::*;
use rustc_query_system::ich::StableHashingContext;
use rustc_span::DUMMY_SP;

/// Top-level HIR node for current owner. This only contains the node for which
/// `HirId::local_id == 0`, and excludes bodies.
///
/// This struct exists to encapsulate all access to the hir_owner query in this module, and to
/// implement HashStable without hashing bodies.
#[derive(Copy, Clone, Debug)]
pub struct Owner<'tcx> {
    node: OwnerNode<'tcx>,
    hash_without_bodies: Fingerprint,
}

impl<'a, 'tcx> HashStable<StableHashingContext<'a>> for Owner<'tcx> {
    #[inline]
    fn hash_stable(&self, hcx: &mut StableHashingContext<'a>, hasher: &mut StableHasher) {
        let Owner { node: _, hash_without_bodies } = self;
        hash_without_bodies.hash_stable(hcx, hasher)
    }
}

/// Gather the LocalDefId for each item-like within a module, including items contained within
/// bodies.  The Ids are in visitor order.  This is used to partition a pass between modules.
#[derive(Debug, HashStable)]
pub struct ModuleItems {
    submodules: Box<[LocalDefId]>,
    items: Box<[ItemId]>,
    trait_items: Box<[TraitItemId]>,
    impl_items: Box<[ImplItemId]>,
    foreign_items: Box<[ForeignItemId]>,
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
        hir.get_module_parent_node(hir.local_def_id_to_hir_id(id))
    };
    providers.hir_crate = |tcx, ()| tcx.untracked_crate;
    providers.crate_hash = map::crate_hash;
    providers.hir_module_items = map::hir_module_items;
    providers.hir_owner = |tcx, id| {
        let owner = tcx.hir_crate(()).owners.get(id)?.as_owner()?;
        let node = owner.node();
        Some(Owner { node, hash_without_bodies: owner.nodes.hash_without_bodies })
    };
    providers.local_def_id_to_hir_id = |tcx, id| {
        let owner = tcx.hir_crate(()).owners[id].map(|_| ());
        match owner {
            MaybeOwner::Owner(_) => HirId::make_owner(id),
            MaybeOwner::Phantom => bug!("No HirId for {:?}", id),
            MaybeOwner::NonOwner(hir_id) => hir_id,
        }
    };
    providers.hir_owner_nodes = |tcx, id| tcx.hir_crate(()).owners[id].map(|i| &i.nodes);
    providers.hir_owner_parent = |tcx, id| {
        // Accessing the def_key is ok since its value is hashed as part of `id`'s DefPathHash.
        let parent = tcx.untracked_resolutions.definitions.def_key(id).parent;
        let parent = parent.map_or(CRATE_HIR_ID, |local_def_index| {
            let def_id = LocalDefId { local_def_index };
            let mut parent_hir_id = tcx.hir().local_def_id_to_hir_id(def_id);
            if let Some(local_id) =
                tcx.hir_crate(()).owners[parent_hir_id.owner].unwrap().parenting.get(&id)
            {
                parent_hir_id.local_id = *local_id;
            }
            parent_hir_id
        });
        parent
    };
    providers.hir_attrs =
        |tcx, id| tcx.hir_crate(()).owners[id].as_owner().map_or(AttributeMap::EMPTY, |o| &o.attrs);
    providers.source_span = |tcx, def_id| tcx.resolutions(()).definitions.def_span(def_id);
    providers.def_span = |tcx, def_id| tcx.hir().span_if_local(def_id).unwrap_or(DUMMY_SP);
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
    providers.opt_def_kind = |tcx, def_id| tcx.hir().opt_def_kind(def_id.expect_local());
    providers.all_local_trait_impls = |tcx, ()| &tcx.resolutions(()).trait_impls;
    providers.expn_that_defined = |tcx, id| {
        let id = id.expect_local();
        tcx.resolutions(()).definitions.expansion_that_defined(id)
    };
    providers.in_scope_traits_map =
        |tcx, id| tcx.hir_crate(()).owners[id].as_owner().map(|owner_info| &owner_info.trait_map);
}
