//! HIR datatypes. See the [rustc dev guide] for more info.
//!
//! [rustc dev guide]: https://rustc-dev-guide.rust-lang.org/hir.html

pub mod map;
pub mod nested_filter;
pub mod place;

use crate::ty::query::Providers;
use crate::ty::{DefIdTree, ImplSubject, TyCtxt};
use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_data_structures::sync::{par_for_each_in, Send, Sync};
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::*;
use rustc_query_system::ich::StableHashingContext;
use rustc_span::{ExpnId, DUMMY_SP};

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
/// bodies. The Ids are in visitor order. This is used to partition a pass between modules.
#[derive(Debug, HashStable, Encodable, Decodable)]
pub struct ModuleItems {
    submodules: Box<[OwnerId]>,
    items: Box<[ItemId]>,
    trait_items: Box<[TraitItemId]>,
    impl_items: Box<[ImplItemId]>,
    foreign_items: Box<[ForeignItemId]>,
    body_owners: Box<[LocalDefId]>,
}

impl ModuleItems {
    pub fn items(&self) -> impl Iterator<Item = ItemId> + '_ {
        self.items.iter().copied()
    }

    pub fn trait_items(&self) -> impl Iterator<Item = TraitItemId> + '_ {
        self.trait_items.iter().copied()
    }

    pub fn impl_items(&self) -> impl Iterator<Item = ImplItemId> + '_ {
        self.impl_items.iter().copied()
    }

    pub fn foreign_items(&self) -> impl Iterator<Item = ForeignItemId> + '_ {
        self.foreign_items.iter().copied()
    }

    pub fn definitions(&self) -> impl Iterator<Item = LocalDefId> + '_ {
        self.items
            .iter()
            .map(|id| id.owner_id.def_id)
            .chain(self.trait_items.iter().map(|id| id.owner_id.def_id))
            .chain(self.impl_items.iter().map(|id| id.owner_id.def_id))
            .chain(self.foreign_items.iter().map(|id| id.owner_id.def_id))
    }

    pub fn par_items(&self, f: impl Fn(ItemId) + Send + Sync) {
        par_for_each_in(&self.items[..], |&id| f(id))
    }

    pub fn par_trait_items(&self, f: impl Fn(TraitItemId) + Send + Sync) {
        par_for_each_in(&self.trait_items[..], |&id| f(id))
    }

    pub fn par_impl_items(&self, f: impl Fn(ImplItemId) + Send + Sync) {
        par_for_each_in(&self.impl_items[..], |&id| f(id))
    }

    pub fn par_foreign_items(&self, f: impl Fn(ForeignItemId) + Send + Sync) {
        par_for_each_in(&self.foreign_items[..], |&id| f(id))
    }
}

impl<'tcx> TyCtxt<'tcx> {
    #[inline(always)]
    pub fn hir(self) -> map::Map<'tcx> {
        map::Map { tcx: self }
    }

    pub fn parent_module(self, id: HirId) -> LocalDefId {
        self.parent_module_from_def_id(id.owner.def_id)
    }

    pub fn impl_subject(self, def_id: DefId) -> ImplSubject<'tcx> {
        self.impl_trait_ref(def_id)
            .map(|t| t.subst_identity())
            .map(ImplSubject::Trait)
            .unwrap_or_else(|| ImplSubject::Inherent(self.type_of(def_id).subst_identity()))
    }
}

pub fn provide(providers: &mut Providers) {
    providers.parent_module_from_def_id = |tcx, id| {
        let hir = tcx.hir();
        hir.get_module_parent_node(hir.local_def_id_to_hir_id(id)).def_id
    };
    providers.hir_crate_items = map::hir_crate_items;
    providers.crate_hash = map::crate_hash;
    providers.hir_module_items = map::hir_module_items;
    providers.hir_owner = |tcx, id| {
        let owner = tcx.hir_crate(()).owners.get(id.def_id)?.as_owner()?;
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
    providers.hir_owner_nodes = |tcx, id| tcx.hir_crate(()).owners[id.def_id].map(|i| &i.nodes);
    providers.hir_owner_parent = |tcx, id| {
        // Accessing the local_parent is ok since its value is hashed as part of `id`'s DefPathHash.
        tcx.opt_local_parent(id.def_id).map_or(CRATE_HIR_ID, |parent| {
            let mut parent_hir_id = tcx.hir().local_def_id_to_hir_id(parent);
            parent_hir_id.local_id =
                tcx.hir_crate(()).owners[parent_hir_id.owner.def_id].unwrap().parenting[&id.def_id];
            parent_hir_id
        })
    };
    providers.hir_attrs = |tcx, id| {
        tcx.hir_crate(()).owners[id.def_id].as_owner().map_or(AttributeMap::EMPTY, |o| &o.attrs)
    };
    providers.def_span = |tcx, def_id| {
        let def_id = def_id.expect_local();
        let hir_id = tcx.hir().local_def_id_to_hir_id(def_id);
        tcx.hir().opt_span(hir_id).unwrap_or(DUMMY_SP)
    };
    providers.def_ident_span = |tcx, def_id| {
        let def_id = def_id.expect_local();
        let hir_id = tcx.hir().local_def_id_to_hir_id(def_id);
        tcx.hir().opt_ident_span(hir_id)
    };
    providers.fn_arg_names = |tcx, id| {
        let hir = tcx.hir();
        let def_id = id.expect_local();
        let hir_id = hir.local_def_id_to_hir_id(def_id);
        if let Some(body_id) = hir.maybe_body_owned_by(def_id) {
            tcx.arena.alloc_from_iter(hir.body_param_names(body_id))
        } else if let Node::TraitItem(&TraitItem {
            kind: TraitItemKind::Fn(_, TraitFn::Required(idents)),
            ..
        })
        | Node::ForeignItem(&ForeignItem {
            kind: ForeignItemKind::Fn(_, idents, _),
            ..
        }) = hir.get(hir_id)
        {
            idents
        } else {
            span_bug!(hir.span(hir_id), "fn_arg_names: unexpected item {:?}", id);
        }
    };
    providers.opt_def_kind = |tcx, def_id| tcx.hir().opt_def_kind(def_id.expect_local());
    providers.all_local_trait_impls = |tcx, ()| &tcx.resolutions(()).trait_impls;
    providers.expn_that_defined = |tcx, id| {
        let id = id.expect_local();
        tcx.resolutions(()).expn_that_defined.get(&id).copied().unwrap_or(ExpnId::root())
    };
    providers.in_scope_traits_map = |tcx, id| {
        tcx.hir_crate(()).owners[id.def_id].as_owner().map(|owner_info| &owner_info.trait_map)
    };
}
