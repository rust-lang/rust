use rustc_data_structures::fx::FxHashMap;
use rustc_hir as hir;
use rustc_hir::def_id::DefId;
use rustc_middle::ty::{self, TyCtxt};

pub fn provide(providers: &mut ty::query::Providers) {
    *providers = ty::query::Providers {
        associated_item,
        associated_item_def_ids,
        associated_items,
        impl_item_implementor_ids,
        ..*providers
    };
}

fn associated_item_def_ids(tcx: TyCtxt<'_>, def_id: DefId) -> &[DefId] {
    let item = tcx.hir().expect_item(def_id.expect_local());
    match item.kind {
        hir::ItemKind::Trait(.., ref trait_item_refs) => tcx.arena.alloc_from_iter(
            trait_item_refs.iter().map(|trait_item_ref| trait_item_ref.id.owner_id.to_def_id()),
        ),
        hir::ItemKind::Impl(ref impl_) => tcx.arena.alloc_from_iter(
            impl_.items.iter().map(|impl_item_ref| impl_item_ref.id.owner_id.to_def_id()),
        ),
        _ => span_bug!(item.span, "associated_item_def_ids: not impl or trait"),
    }
}

fn associated_items(tcx: TyCtxt<'_>, def_id: DefId) -> ty::AssocItems {
    if tcx.is_trait_alias(def_id) {
        ty::AssocItems::new(Vec::new())
    } else {
        let items = tcx.associated_item_def_ids(def_id).iter().map(|did| tcx.associated_item(*did));
        ty::AssocItems::new(items)
    }
}

fn impl_item_implementor_ids(tcx: TyCtxt<'_>, impl_id: DefId) -> FxHashMap<DefId, DefId> {
    tcx.associated_items(impl_id)
        .in_definition_order()
        .filter_map(|item| item.trait_item_def_id.map(|trait_item| (trait_item, item.def_id)))
        .collect()
}

fn associated_item(tcx: TyCtxt<'_>, def_id: DefId) -> ty::AssocItem {
    let id = tcx.hir().local_def_id_to_hir_id(def_id.expect_local());
    let parent_def_id = tcx.hir().get_parent_item(id);
    let parent_item = tcx.hir().expect_item(parent_def_id.def_id);
    match parent_item.kind {
        hir::ItemKind::Impl(ref impl_) => {
            if let Some(impl_item_ref) =
                impl_.items.iter().find(|i| i.id.owner_id.to_def_id() == def_id)
            {
                let assoc_item = associated_item_from_impl_item_ref(impl_item_ref);
                debug_assert_eq!(assoc_item.def_id, def_id);
                return assoc_item;
            }
        }

        hir::ItemKind::Trait(.., ref trait_item_refs) => {
            if let Some(trait_item_ref) =
                trait_item_refs.iter().find(|i| i.id.owner_id.to_def_id() == def_id)
            {
                let assoc_item = associated_item_from_trait_item_ref(trait_item_ref);
                debug_assert_eq!(assoc_item.def_id, def_id);
                return assoc_item;
            }
        }

        _ => {}
    }

    span_bug!(
        parent_item.span,
        "unexpected parent of trait or impl item or item not found: {:?}",
        parent_item.kind
    )
}

fn associated_item_from_trait_item_ref(trait_item_ref: &hir::TraitItemRef) -> ty::AssocItem {
    let owner_id = trait_item_ref.id.owner_id;
    let (kind, has_self) = match trait_item_ref.kind {
        hir::AssocItemKind::Const => (ty::AssocKind::Const, false),
        hir::AssocItemKind::Fn { has_self } => (ty::AssocKind::Fn, has_self),
        hir::AssocItemKind::Type => (ty::AssocKind::Type, false),
    };

    ty::AssocItem {
        name: trait_item_ref.ident.name,
        kind,
        def_id: owner_id.to_def_id(),
        trait_item_def_id: Some(owner_id.to_def_id()),
        container: ty::TraitContainer,
        fn_has_self_parameter: has_self,
    }
}

fn associated_item_from_impl_item_ref(impl_item_ref: &hir::ImplItemRef) -> ty::AssocItem {
    let def_id = impl_item_ref.id.owner_id;
    let (kind, has_self) = match impl_item_ref.kind {
        hir::AssocItemKind::Const => (ty::AssocKind::Const, false),
        hir::AssocItemKind::Fn { has_self } => (ty::AssocKind::Fn, has_self),
        hir::AssocItemKind::Type => (ty::AssocKind::Type, false),
    };

    ty::AssocItem {
        name: impl_item_ref.ident.name,
        kind,
        def_id: def_id.to_def_id(),
        trait_item_def_id: impl_item_ref.trait_item_def_id,
        container: ty::ImplContainer,
        fn_has_self_parameter: has_self,
    }
}
