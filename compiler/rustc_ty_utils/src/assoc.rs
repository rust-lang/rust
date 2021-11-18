use rustc_data_structures::fx::FxHashMap;
use rustc_errors::struct_span_err;
use rustc_hir as hir;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_middle::ty::{self, TyCtxt, TypeFoldable};

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
    let parent_id = tcx.hir().get_parent_item(id);
    let parent_def_id = tcx.hir().local_def_id(parent_id);
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
        ident: trait_item_ref.ident,
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

    let trait_item_def_id = impl_item_base_id(tcx, parent_def_id, impl_item_ref);

    ty::AssocItem {
        ident: impl_item_ref.ident,
        kind,
        vis: tcx.visibility(def_id),
        defaultness: impl_item_ref.defaultness,
        def_id: def_id.to_def_id(),
        trait_item_def_id,
        container: ty::ImplContainer(parent_def_id.to_def_id()),
        fn_has_self_parameter: has_self,
    }
}

fn impl_item_base_id<'tcx>(
    tcx: TyCtxt<'tcx>,
    parent_def_id: LocalDefId,
    impl_item: &hir::ImplItemRef,
) -> Option<DefId> {
    let impl_trait_ref = tcx.impl_trait_ref(parent_def_id)?;

    // If the trait reference itself is erroneous (so the compilation is going
    // to fail), skip checking the items here -- the `impl_item` table in `tcx`
    // isn't populated for such impls.
    if impl_trait_ref.references_error() {
        return None;
    }

    // Locate trait items
    let associated_items = tcx.associated_items(impl_trait_ref.def_id);

    // Match item against trait
    let mut items = associated_items.filter_by_name(tcx, impl_item.ident, impl_trait_ref.def_id);

    let mut trait_item = items.next()?;

    let is_compatible = |ty: &&ty::AssocItem| match (ty.kind, &impl_item.kind) {
        (ty::AssocKind::Const, hir::AssocItemKind::Const) => true,
        (ty::AssocKind::Fn, hir::AssocItemKind::Fn { .. }) => true,
        (ty::AssocKind::Type, hir::AssocItemKind::Type) => true,
        _ => false,
    };

    // If we don't have a compatible item, we'll use the first one whose name matches
    // to report an error.
    let mut compatible_kind = is_compatible(&trait_item);

    if !compatible_kind {
        if let Some(ty_trait_item) = items.find(is_compatible) {
            compatible_kind = true;
            trait_item = ty_trait_item;
        }
    }

    if compatible_kind {
        Some(trait_item.def_id)
    } else {
        report_mismatch_error(tcx, trait_item.def_id, impl_trait_ref, impl_item);
        None
    }
}

#[inline(never)]
#[cold]
fn report_mismatch_error<'tcx>(
    tcx: TyCtxt<'tcx>,
    trait_item_def_id: DefId,
    impl_trait_ref: ty::TraitRef<'tcx>,
    impl_item: &hir::ImplItemRef,
) {
    let mut err = match impl_item.kind {
        hir::AssocItemKind::Const => {
            // Find associated const definition.
            struct_span_err!(
                tcx.sess,
                impl_item.span,
                E0323,
                "item `{}` is an associated const, which doesn't match its trait `{}`",
                impl_item.ident,
                impl_trait_ref.print_only_trait_path()
            )
        }

        hir::AssocItemKind::Fn { .. } => {
            struct_span_err!(
                tcx.sess,
                impl_item.span,
                E0324,
                "item `{}` is an associated method, which doesn't match its trait `{}`",
                impl_item.ident,
                impl_trait_ref.print_only_trait_path()
            )
        }

        hir::AssocItemKind::Type => {
            struct_span_err!(
                tcx.sess,
                impl_item.span,
                E0325,
                "item `{}` is an associated type, which doesn't match its trait `{}`",
                impl_item.ident,
                impl_trait_ref.print_only_trait_path()
            )
        }
    };

    err.span_label(impl_item.span, "does not match trait");
    if let Some(trait_span) = tcx.hir().span_if_local(trait_item_def_id) {
        err.span_label(trait_span, "item in trait");
    }
    err.emit();
}
