//! HIR datatypes. See the [rustc dev guide] for more info.
//!
//! [rustc dev guide]: https://rustc-dev-guide.rust-lang.org/hir.html

pub mod map;
pub mod nested_filter;
pub mod place;

use crate::query::Providers;
use crate::ty::{EarlyBinder, ImplSubject, TyCtxt};
use rustc_data_structures::sync::{try_par_for_each_in, DynSend, DynSync};
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{DefId, LocalDefId, LocalModDefId};
use rustc_hir::*;
use rustc_span::{ErrorGuaranteed, ExpnId, DUMMY_SP};

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

    pub fn owners(&self) -> impl Iterator<Item = OwnerId> + '_ {
        self.items
            .iter()
            .map(|id| id.owner_id)
            .chain(self.trait_items.iter().map(|id| id.owner_id))
            .chain(self.impl_items.iter().map(|id| id.owner_id))
            .chain(self.foreign_items.iter().map(|id| id.owner_id))
    }

    pub fn definitions(&self) -> impl Iterator<Item = LocalDefId> + '_ {
        self.owners().map(|id| id.def_id)
    }

    pub fn par_items(
        &self,
        f: impl Fn(ItemId) -> Result<(), ErrorGuaranteed> + DynSend + DynSync,
    ) -> Result<(), ErrorGuaranteed> {
        try_par_for_each_in(&self.items[..], |&id| f(id))
    }

    pub fn par_trait_items(
        &self,
        f: impl Fn(TraitItemId) -> Result<(), ErrorGuaranteed> + DynSend + DynSync,
    ) -> Result<(), ErrorGuaranteed> {
        try_par_for_each_in(&self.trait_items[..], |&id| f(id))
    }

    pub fn par_impl_items(
        &self,
        f: impl Fn(ImplItemId) -> Result<(), ErrorGuaranteed> + DynSend + DynSync,
    ) -> Result<(), ErrorGuaranteed> {
        try_par_for_each_in(&self.impl_items[..], |&id| f(id))
    }

    pub fn par_foreign_items(
        &self,
        f: impl Fn(ForeignItemId) -> Result<(), ErrorGuaranteed> + DynSend + DynSync,
    ) -> Result<(), ErrorGuaranteed> {
        try_par_for_each_in(&self.foreign_items[..], |&id| f(id))
    }
}

impl<'tcx> TyCtxt<'tcx> {
    #[inline(always)]
    pub fn hir(self) -> map::Map<'tcx> {
        map::Map { tcx: self }
    }

    pub fn parent_module(self, id: HirId) -> LocalModDefId {
        if !id.is_owner() && self.def_kind(id.owner) == DefKind::Mod {
            LocalModDefId::new_unchecked(id.owner.def_id)
        } else {
            self.parent_module_from_def_id(id.owner.def_id)
        }
    }

    pub fn parent_module_from_def_id(self, mut id: LocalDefId) -> LocalModDefId {
        while let Some(parent) = self.opt_local_parent(id) {
            id = parent;
            if self.def_kind(id) == DefKind::Mod {
                break;
            }
        }
        LocalModDefId::new_unchecked(id)
    }

    pub fn impl_subject(self, def_id: DefId) -> EarlyBinder<ImplSubject<'tcx>> {
        match self.impl_trait_ref(def_id) {
            Some(t) => t.map_bound(ImplSubject::Trait),
            None => self.type_of(def_id).map_bound(ImplSubject::Inherent),
        }
    }

    /// Returns `true` if this is a foreign item (i.e., linked via `extern { ... }`).
    pub fn is_foreign_item(self, def_id: impl Into<DefId>) -> bool {
        self.opt_parent(def_id.into())
            .is_some_and(|parent| matches!(self.def_kind(parent), DefKind::ForeignMod))
    }
}

pub fn provide(providers: &mut Providers) {
    providers.hir_crate_items = map::hir_crate_items;
    providers.crate_hash = map::crate_hash;
    providers.hir_module_items = map::hir_module_items;
    providers.opt_local_def_id_to_hir_id = |tcx, def_id| {
        Some(match tcx.hir_crate(()).owners[def_id] {
            MaybeOwner::Owner(_) => HirId::make_owner(def_id),
            MaybeOwner::NonOwner(hir_id) => hir_id,
            MaybeOwner::Phantom => bug!("No HirId for {:?}", def_id),
        })
    };
    providers.opt_hir_owner_nodes =
        |tcx, id| tcx.hir_crate(()).owners.get(id)?.as_owner().map(|i| &i.nodes);
    providers.hir_owner_parent = |tcx, id| {
        // Accessing the local_parent is ok since its value is hashed as part of `id`'s DefPathHash.
        tcx.opt_local_parent(id.def_id).map_or(CRATE_HIR_ID, |parent| {
            let mut parent_hir_id = tcx.local_def_id_to_hir_id(parent);
            parent_hir_id.local_id =
                tcx.hir_crate(()).owners[parent_hir_id.owner.def_id].unwrap().parenting[&id.def_id];
            parent_hir_id
        })
    };
    providers.hir_attrs = |tcx, id| {
        tcx.hir_crate(()).owners[id.def_id].as_owner().map_or(AttributeMap::EMPTY, |o| &o.attrs)
    };
    providers.def_span = |tcx, def_id| {
        let hir_id = tcx.local_def_id_to_hir_id(def_id);
        tcx.hir().opt_span(hir_id).unwrap_or(DUMMY_SP)
    };
    providers.def_ident_span = |tcx, def_id| {
        let hir_id = tcx.local_def_id_to_hir_id(def_id);
        tcx.hir().opt_ident_span(hir_id)
    };
    providers.fn_arg_names = |tcx, def_id| {
        let hir = tcx.hir();
        let hir_id = tcx.local_def_id_to_hir_id(def_id);
        if let Some(body_id) = hir.maybe_body_owned_by(def_id) {
            tcx.arena.alloc_from_iter(hir.body_param_names(body_id))
        } else if let Node::TraitItem(&TraitItem {
            kind: TraitItemKind::Fn(_, TraitFn::Required(idents)),
            ..
        })
        | Node::ForeignItem(&ForeignItem {
            kind: ForeignItemKind::Fn(_, idents, _),
            ..
        }) = tcx.hir_node(hir_id)
        {
            idents
        } else {
            span_bug!(hir.span(hir_id), "fn_arg_names: unexpected item {:?}", def_id);
        }
    };
    providers.all_local_trait_impls = |tcx, ()| &tcx.resolutions(()).trait_impls;
    providers.expn_that_defined =
        |tcx, id| tcx.resolutions(()).expn_that_defined.get(&id).copied().unwrap_or(ExpnId::root());
    providers.in_scope_traits_map = |tcx, id| {
        tcx.hir_crate(()).owners[id.def_id].as_owner().map(|owner_info| &owner_info.trait_map)
    };
}
