//! HIR datatypes. See the [rustc dev guide] for more info.
//!
//! [rustc dev guide]: https://rustc-dev-guide.rust-lang.org/hir.html

pub mod map;
pub mod nested_filter;
pub mod place;

use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::sorted_map::SortedMap;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_data_structures::sync::{DynSend, DynSync, try_par_for_each_in};
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{DefId, LocalDefId, LocalModDefId};
use rustc_hir::lints::DelayedLint;
use rustc_hir::*;
use rustc_macros::{Decodable, Encodable, HashStable};
use rustc_span::{ErrorGuaranteed, ExpnId, Span};

use crate::query::Providers;
use crate::ty::TyCtxt;

/// Gather the LocalDefId for each item-like within a module, including items contained within
/// bodies. The Ids are in visitor order. This is used to partition a pass between modules.
#[derive(Debug, HashStable, Encodable, Decodable)]
pub struct ModuleItems {
    /// Whether this represents the whole crate, in which case we need to add `CRATE_OWNER_ID` to
    /// the iterators if we want to account for the crate root.
    add_root: bool,
    submodules: Box<[OwnerId]>,
    free_items: Box<[ItemId]>,
    trait_items: Box<[TraitItemId]>,
    impl_items: Box<[ImplItemId]>,
    foreign_items: Box<[ForeignItemId]>,
    opaques: Box<[LocalDefId]>,
    body_owners: Box<[LocalDefId]>,
    nested_bodies: Box<[LocalDefId]>,
    // only filled with hir_crate_items, not with hir_module_items
    delayed_lint_items: Box<[OwnerId]>,
}

impl ModuleItems {
    /// Returns all non-associated locally defined items in all modules.
    ///
    /// Note that this does *not* include associated items of `impl` blocks! It also does not
    /// include foreign items. If you want to e.g. get all functions, use `definitions()` below.
    ///
    /// However, this does include the `impl` blocks themselves.
    pub fn free_items(&self) -> impl Iterator<Item = ItemId> {
        self.free_items.iter().copied()
    }

    pub fn trait_items(&self) -> impl Iterator<Item = TraitItemId> {
        self.trait_items.iter().copied()
    }

    pub fn delayed_lint_items(&self) -> impl Iterator<Item = OwnerId> {
        self.delayed_lint_items.iter().copied()
    }

    /// Returns all items that are associated with some `impl` block (both inherent and trait impl
    /// blocks).
    pub fn impl_items(&self) -> impl Iterator<Item = ImplItemId> {
        self.impl_items.iter().copied()
    }

    pub fn foreign_items(&self) -> impl Iterator<Item = ForeignItemId> {
        self.foreign_items.iter().copied()
    }

    pub fn owners(&self) -> impl Iterator<Item = OwnerId> {
        self.add_root
            .then_some(CRATE_OWNER_ID)
            .into_iter()
            .chain(self.free_items.iter().map(|id| id.owner_id))
            .chain(self.trait_items.iter().map(|id| id.owner_id))
            .chain(self.impl_items.iter().map(|id| id.owner_id))
            .chain(self.foreign_items.iter().map(|id| id.owner_id))
    }

    pub fn opaques(&self) -> impl Iterator<Item = LocalDefId> {
        self.opaques.iter().copied()
    }

    /// Closures and inline consts
    pub fn nested_bodies(&self) -> impl Iterator<Item = LocalDefId> {
        self.nested_bodies.iter().copied()
    }

    pub fn definitions(&self) -> impl Iterator<Item = LocalDefId> {
        self.owners().map(|id| id.def_id)
    }

    /// Closures and inline consts
    pub fn par_nested_bodies(
        &self,
        f: impl Fn(LocalDefId) -> Result<(), ErrorGuaranteed> + DynSend + DynSync,
    ) -> Result<(), ErrorGuaranteed> {
        try_par_for_each_in(&self.nested_bodies[..], |&&id| f(id))
    }

    pub fn par_items(
        &self,
        f: impl Fn(ItemId) -> Result<(), ErrorGuaranteed> + DynSend + DynSync,
    ) -> Result<(), ErrorGuaranteed> {
        try_par_for_each_in(&self.free_items[..], |&&id| f(id))
    }

    pub fn par_trait_items(
        &self,
        f: impl Fn(TraitItemId) -> Result<(), ErrorGuaranteed> + DynSend + DynSync,
    ) -> Result<(), ErrorGuaranteed> {
        try_par_for_each_in(&self.trait_items[..], |&&id| f(id))
    }

    pub fn par_impl_items(
        &self,
        f: impl Fn(ImplItemId) -> Result<(), ErrorGuaranteed> + DynSend + DynSync,
    ) -> Result<(), ErrorGuaranteed> {
        try_par_for_each_in(&self.impl_items[..], |&&id| f(id))
    }

    pub fn par_foreign_items(
        &self,
        f: impl Fn(ForeignItemId) -> Result<(), ErrorGuaranteed> + DynSend + DynSync,
    ) -> Result<(), ErrorGuaranteed> {
        try_par_for_each_in(&self.foreign_items[..], |&&id| f(id))
    }

    pub fn par_opaques(
        &self,
        f: impl Fn(LocalDefId) -> Result<(), ErrorGuaranteed> + DynSend + DynSync,
    ) -> Result<(), ErrorGuaranteed> {
        try_par_for_each_in(&self.opaques[..], |&&id| f(id))
    }
}

impl<'tcx> TyCtxt<'tcx> {
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

    /// Returns `true` if this is a foreign item (i.e., linked via `extern { ... }`).
    pub fn is_foreign_item(self, def_id: impl Into<DefId>) -> bool {
        self.opt_parent(def_id.into())
            .is_some_and(|parent| matches!(self.def_kind(parent), DefKind::ForeignMod))
    }

    pub fn hash_owner_nodes(
        self,
        node: OwnerNode<'_>,
        bodies: &SortedMap<ItemLocalId, &Body<'_>>,
        attrs: &SortedMap<ItemLocalId, &[Attribute]>,
        delayed_lints: &[DelayedLint],
        define_opaque: Option<&[(Span, LocalDefId)]>,
    ) -> Hashes {
        if !self.needs_crate_hash() {
            return Hashes {
                opt_hash_including_bodies: None,
                attrs_hash: None,
                delayed_lints_hash: None,
            };
        }

        self.with_stable_hashing_context(|mut hcx| {
            let mut stable_hasher = StableHasher::new();
            node.hash_stable(&mut hcx, &mut stable_hasher);
            // Bodies are stored out of line, so we need to pull them explicitly in the hash.
            bodies.hash_stable(&mut hcx, &mut stable_hasher);
            let h1 = stable_hasher.finish();

            let mut stable_hasher = StableHasher::new();
            attrs.hash_stable(&mut hcx, &mut stable_hasher);

            // Hash the defined opaque types, which are not present in the attrs.
            define_opaque.hash_stable(&mut hcx, &mut stable_hasher);

            let h2 = stable_hasher.finish();

            // hash lints emitted during ast lowering
            let mut stable_hasher = StableHasher::new();
            delayed_lints.hash_stable(&mut hcx, &mut stable_hasher);
            let h3 = stable_hasher.finish();

            Hashes {
                opt_hash_including_bodies: Some(h1),
                attrs_hash: Some(h2),
                delayed_lints_hash: Some(h3),
            }
        })
    }
}

/// Hashes computed by [`TyCtxt::hash_owner_nodes`] if necessary.
#[derive(Clone, Copy, Debug)]
pub struct Hashes {
    pub opt_hash_including_bodies: Option<Fingerprint>,
    pub attrs_hash: Option<Fingerprint>,
    pub delayed_lints_hash: Option<Fingerprint>,
}

pub fn provide(providers: &mut Providers) {
    providers.hir_crate_items = map::hir_crate_items;
    providers.crate_hash = map::crate_hash;
    providers.hir_module_items = map::hir_module_items;
    providers.local_def_id_to_hir_id = |tcx, def_id| match tcx.hir_crate(()).owners[def_id] {
        MaybeOwner::Owner(_) => HirId::make_owner(def_id),
        MaybeOwner::NonOwner(hir_id) => hir_id,
        MaybeOwner::Phantom => bug!("No HirId for {:?}", def_id),
    };
    providers.opt_hir_owner_nodes =
        |tcx, id| tcx.hir_crate(()).owners.get(id)?.as_owner().map(|i| &i.nodes);
    providers.hir_owner_parent = |tcx, owner_id| {
        tcx.opt_local_parent(owner_id.def_id).map_or(CRATE_HIR_ID, |parent_def_id| {
            let parent_owner_id = tcx.local_def_id_to_hir_id(parent_def_id).owner;
            HirId {
                owner: parent_owner_id,
                local_id: tcx.hir_crate(()).owners[parent_owner_id.def_id]
                    .unwrap()
                    .parenting
                    .get(&owner_id.def_id)
                    .copied()
                    .unwrap_or(ItemLocalId::ZERO),
            }
        })
    };
    providers.hir_attr_map = |tcx, id| {
        tcx.hir_crate(()).owners[id.def_id].as_owner().map_or(AttributeMap::EMPTY, |o| &o.attrs)
    };
    providers.opt_ast_lowering_delayed_lints =
        |tcx, id| tcx.hir_crate(()).owners[id.def_id].as_owner().map(|o| &o.delayed_lints);
    providers.def_span = |tcx, def_id| tcx.hir_span(tcx.local_def_id_to_hir_id(def_id));
    providers.def_ident_span = |tcx, def_id| {
        let hir_id = tcx.local_def_id_to_hir_id(def_id);
        tcx.hir_opt_ident_span(hir_id)
    };
    providers.ty_span = |tcx, def_id| {
        let node = tcx.hir_node_by_def_id(def_id);
        match node.ty() {
            Some(ty) => ty.span,
            None => bug!("{def_id:?} doesn't have a type: {node:#?}"),
        }
    };
    providers.fn_arg_idents = |tcx, def_id| {
        let node = tcx.hir_node_by_def_id(def_id);
        if let Some(body_id) = node.body_id() {
            tcx.arena.alloc_from_iter(tcx.hir_body_param_idents(body_id))
        } else if let Node::TraitItem(&TraitItem {
            kind: TraitItemKind::Fn(_, TraitFn::Required(idents)),
            ..
        })
        | Node::ForeignItem(&ForeignItem {
            kind: ForeignItemKind::Fn(_, idents, _),
            ..
        }) = node
        {
            idents
        } else {
            span_bug!(
                tcx.hir_span(tcx.local_def_id_to_hir_id(def_id)),
                "fn_arg_idents: unexpected item {:?}",
                def_id
            );
        }
    };
    providers.all_local_trait_impls = |tcx, ()| &tcx.resolutions(()).trait_impls;
    providers.local_trait_impls =
        |tcx, trait_id| tcx.resolutions(()).trait_impls.get(&trait_id).map_or(&[], |xs| &xs[..]);
    providers.expn_that_defined =
        |tcx, id| tcx.resolutions(()).expn_that_defined.get(&id).copied().unwrap_or(ExpnId::root());
    providers.in_scope_traits_map = |tcx, id| {
        tcx.hir_crate(()).owners[id.def_id].as_owner().map(|owner_info| &owner_info.trait_map)
    };
}
