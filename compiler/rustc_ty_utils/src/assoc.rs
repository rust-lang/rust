use rustc_data_structures::fx::FxHashMap;
use rustc_hir as hir;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_middle::ty::{self, TyCtxt};

pub fn provide(providers: &mut ty::query::Providers) {
    *providers = ty::query::Providers {
        associated_item,
        associated_item_def_ids,
        associated_items,
        impl_item_implementor_ids,
        trait_of_item,
        ..*providers
    };
}

fn associated_item_def_ids(tcx: TyCtxt<'_>, def_id: DefId) -> &[DefId] {
    let item = tcx.hir().expect_item(def_id.expect_local());
    match item.kind {
        hir::ItemKind::Trait(.., ref trait_item_refs) => tcx.arena.alloc_from_iter(
            trait_item_refs.iter().map(|trait_item_ref| trait_item_ref.id.def_id.to_def_id()),
        ),
        hir::ItemKind::Impl(ref impl_) => tcx.arena.alloc_from_iter(
            impl_.items.iter().map(|impl_item_ref| impl_item_ref.id.def_id.to_def_id()),
        ),
        hir::ItemKind::TraitAlias(..) => &[],
        _ => span_bug!(item.span, "associated_item_def_ids: not impl or trait"),
    }
}

fn associated_items(tcx: TyCtxt<'_>, def_id: DefId) -> ty::AssocItems<'_> {
    let items = tcx.associated_item_def_ids(def_id).iter().map(|did| tcx.associated_item(*did));
    ty::AssocItems::new(items)
}

fn impl_item_implementor_ids(tcx: TyCtxt<'_>, impl_id: DefId) -> FxHashMap<DefId, DefId> {
    tcx.associated_items(impl_id)
        .in_definition_order()
        .filter_map(|item| item.trait_item_def_id.map(|trait_item| (trait_item, item.def_id)))
        .collect()
}

/// If the given `DefId` describes an item belonging to a trait,
/// returns the `DefId` of the trait that the trait item belongs to;
/// otherwise, returns `None`.
fn trait_of_item(tcx: TyCtxt<'_>, def_id: DefId) -> Option<DefId> {
    tcx.opt_associated_item(def_id).and_then(|associated_item| match associated_item.container {
        ty::TraitContainer(def_id) => Some(def_id),
        ty::ImplContainer(_) => None,
    })
}

fn associated_item(tcx: TyCtxt<'_>, def_id: DefId) -> ty::AssocItem {
    let id = tcx.hir().local_def_id_to_hir_id(def_id.expect_local());
    let parent_def_id = tcx.hir().get_parent_item(id);
    let parent_item = tcx.hir().expect_item(parent_def_id);
    match parent_item.kind {
        hir::ItemKind::Impl(ref impl_) => {
            if let Some(impl_item_ref) =
                impl_.items.iter().find(|i| i.id.def_id.to_def_id() == def_id)
            {
                let assoc_item =
                    associated_item_from_impl_item_ref(tcx, parent_def_id, impl_item_ref);
                debug_assert_eq!(assoc_item.def_id, def_id);
                return assoc_item;
            }
        }

        hir::ItemKind::Trait(.., ref trait_item_refs) => {
            if let Some(trait_item_ref) =
                trait_item_refs.iter().find(|i| i.id.def_id.to_def_id() == def_id)
            {
                let assoc_item =
                    associated_item_from_trait_item_ref(tcx, parent_def_id, trait_item_ref);
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

fn associated_item_from_trait_item_ref(
    tcx: TyCtxt<'_>,
    parent_def_id: LocalDefId,
    trait_item_ref: &hir::TraitItemRef,
) -> ty::AssocItem {
    let def_id = trait_item_ref.id.def_id;
    let (kind, has_self) = match trait_item_ref.kind {
        hir::AssocItemKind::Const => (ty::AssocKind::Const, false),
        hir::AssocItemKind::Fn { has_self } => (ty::AssocKind::Fn, has_self),
        hir::AssocItemKind::Type => (ty::AssocKind::Type, false),
    };

    ty::AssocItem {
        name: trait_item_ref.ident.name,
        kind,
        vis: tcx.visibility(def_id),
        defaultness: trait_item_ref.defaultness,
        def_id: def_id.to_def_id(),
        trait_item_def_id: Some(def_id.to_def_id()),
        container: ty::TraitContainer(parent_def_id.to_def_id()),
        fn_has_self_parameter: has_self,
    }
}

fn associated_item_from_impl_item_ref(
    tcx: TyCtxt<'_>,
    parent_def_id: LocalDefId,
    impl_item_ref: &hir::ImplItemRef,
) -> ty::AssocItem {
    let def_id = impl_item_ref.id.def_id;
    let (kind, has_self) = match impl_item_ref.kind {
        hir::AssocItemKind::Const => (ty::AssocKind::Const, false),
        hir::AssocItemKind::Fn { has_self } => (ty::AssocKind::Fn, has_self),
        hir::AssocItemKind::Type => (ty::AssocKind::Type, false),
    };

    ty::AssocItem {
        name: impl_item_ref.ident.name,
        kind,
        vis: tcx.visibility(def_id),
        defaultness: impl_item_ref.defaultness,
        def_id: def_id.to_def_id(),
        trait_item_def_id: impl_item_ref.trait_item_def_id,
        container: ty::ImplContainer(parent_def_id.to_def_id()),
        fn_has_self_parameter: has_self,
    }
}
