use rustc_data_structures::fx::FxHashMap;
use rustc_hir as hir;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::definitions::DefPathData;
use rustc_hir::intravisit::{self, Visitor};
use rustc_middle::ty::{self, DefIdTree, ImplTraitInTraitData, InternalSubsts, TyCtxt};
use rustc_span::symbol::kw;

pub fn provide(providers: &mut ty::query::Providers) {
    *providers = ty::query::Providers {
        associated_item,
        associated_item_def_ids,
        associated_items,
        associated_items_for_impl_trait_in_trait,
        associated_item_for_impl_trait_in_trait,
        impl_item_implementor_ids,
        ..*providers
    };
}

fn associated_item_def_ids(tcx: TyCtxt<'_>, def_id: DefId) -> &[DefId] {
    let item = tcx.hir().expect_item(def_id.expect_local());
    match item.kind {
        hir::ItemKind::Trait(.., ref trait_item_refs) => {
            if tcx.sess.opts.unstable_opts.lower_impl_trait_in_trait_to_assoc_ty {
                // We collect RPITITs for each trait method's return type and create a
                // corresponding associated item using associated_items_for_impl_trait_in_trait
                // query.
                tcx.arena.alloc_from_iter(
                    trait_item_refs
                        .iter()
                        .map(|trait_item_ref| trait_item_ref.id.owner_id.to_def_id())
                        .chain(
                            trait_item_refs
                                .iter()
                                .filter(|trait_item_ref| {
                                    matches!(trait_item_ref.kind, hir::AssocItemKind::Fn { .. })
                                })
                                .flat_map(|trait_item_ref| {
                                    let trait_fn_def_id =
                                        trait_item_ref.id.owner_id.def_id.to_def_id();
                                    tcx.associated_items_for_impl_trait_in_trait(trait_fn_def_id)
                                })
                                .map(|def_id| *def_id),
                        ),
                )
            } else {
                tcx.arena.alloc_from_iter(
                    trait_item_refs
                        .iter()
                        .map(|trait_item_ref| trait_item_ref.id.owner_id.to_def_id()),
                )
            }
        }
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

/// Given an `fn_def_id` of a trait or of an impl that implements a given trait:
/// if `fn_def_id` is the def id of a function defined inside a trait, then it creates and returns
/// the associated items that correspond to each impl trait in return position for that trait.
/// if `fn_def_id` is the def id of a function defined inside an impl that implements a trait, then it
/// creates and returns the associated items that correspond to each impl trait in return position
/// of the implemented trait.
fn associated_items_for_impl_trait_in_trait(tcx: TyCtxt<'_>, fn_def_id: DefId) -> &'_ [DefId] {
    let parent_def_id = tcx.parent(fn_def_id);

    match tcx.def_kind(parent_def_id) {
        DefKind::Trait => {
            struct RPITVisitor {
                rpits: Vec<LocalDefId>,
            }

            impl<'v> Visitor<'v> for RPITVisitor {
                fn visit_ty(&mut self, ty: &'v hir::Ty<'v>) {
                    if let hir::TyKind::OpaqueDef(item_id, _, _) = ty.kind {
                        self.rpits.push(item_id.owner_id.def_id)
                    }
                    intravisit::walk_ty(self, ty)
                }
            }

            let mut visitor = RPITVisitor { rpits: Vec::new() };

            if let Some(output) = tcx.hir().get_fn_output(fn_def_id.expect_local()) {
                visitor.visit_fn_ret_ty(output);

                tcx.arena.alloc_from_iter(visitor.rpits.iter().map(|opaque_ty_def_id| {
                    tcx.associated_item_for_impl_trait_in_trait(opaque_ty_def_id).to_def_id()
                }))
            } else {
                &[]
            }
        }

        DefKind::Impl { .. } => {
            let Some(trait_fn_def_id) = tcx.associated_item(fn_def_id).trait_item_def_id else { return &[] };

            tcx.arena.alloc_from_iter(
                tcx.associated_items_for_impl_trait_in_trait(trait_fn_def_id).iter().map(
                    move |trait_assoc_def_id| {
                        impl_associated_item_for_impl_trait_in_trait(
                            tcx,
                            trait_assoc_def_id.expect_local(),
                            fn_def_id.expect_local(),
                        )
                        .to_def_id()
                    },
                ),
            )
        }

        def_kind => bug!(
            "associated_items_for_impl_trait_in_trait: {:?} should be Trait or Impl but is {:?}",
            parent_def_id,
            def_kind
        ),
    }
}

/// Given an `opaque_ty_def_id` corresponding to an impl trait in trait, create and return the
/// corresponding associated item.
fn associated_item_for_impl_trait_in_trait(
    tcx: TyCtxt<'_>,
    opaque_ty_def_id: LocalDefId,
) -> LocalDefId {
    let fn_def_id = tcx.impl_trait_in_trait_parent(opaque_ty_def_id.to_def_id());
    let trait_def_id = tcx.parent(fn_def_id);
    assert_eq!(tcx.def_kind(trait_def_id), DefKind::Trait);

    let span = tcx.def_span(opaque_ty_def_id);
    let trait_assoc_ty =
        tcx.at(span).create_def(trait_def_id.expect_local(), DefPathData::ImplTraitAssocTy);

    let local_def_id = trait_assoc_ty.def_id();
    let def_id = local_def_id.to_def_id();

    trait_assoc_ty.opt_def_kind(Some(DefKind::AssocTy));

    // There's no HIR associated with this new synthesized `def_id`, so feed
    // `opt_local_def_id_to_hir_id` with `None`.
    trait_assoc_ty.opt_local_def_id_to_hir_id(None);

    // Copy span of the opaque.
    trait_assoc_ty.def_ident_span(Some(span));

    // Add the def_id of the function and opaque that generated this synthesized associated type.
    trait_assoc_ty.opt_rpitit_info(Some(ImplTraitInTraitData::Trait {
        fn_def_id,
        opaque_def_id: opaque_ty_def_id.to_def_id(),
    }));

    trait_assoc_ty.associated_item(ty::AssocItem {
        name: kw::Empty,
        kind: ty::AssocKind::Type,
        def_id,
        trait_item_def_id: None,
        container: ty::TraitContainer,
        fn_has_self_parameter: false,
    });

    // Copy visility of the containing function.
    trait_assoc_ty.visibility(tcx.visibility(fn_def_id));

    // Copy impl_defaultness of the containing function.
    trait_assoc_ty.impl_defaultness(tcx.impl_defaultness(fn_def_id));

    // Copy type_of of the opaque.
    trait_assoc_ty.type_of(ty::EarlyBinder(tcx.mk_opaque(
        opaque_ty_def_id.to_def_id(),
        InternalSubsts::identity_for_item(tcx, opaque_ty_def_id.to_def_id()),
    )));

    // Copy generics_of of the opaque.
    trait_assoc_ty.generics_of(tcx.generics_of(opaque_ty_def_id).clone());

    // There are no predicates for the synthesized associated type.
    trait_assoc_ty.explicit_predicates_of(ty::GenericPredicates {
        parent: Some(trait_def_id),
        predicates: &[],
    });

    // There are no inferred outlives for the synthesized associated type.
    trait_assoc_ty.inferred_outlives_of(&[]);

    // FIXME implement this.
    trait_assoc_ty.explicit_item_bounds(&[]);

    local_def_id
}

/// Given an `trait_assoc_def_id` that corresponds to a previously synthesized impl trait in trait
/// into an associated type and an `impl_def_id` corresponding to an impl block, create and return
/// the corresponding associated item inside the impl block.
fn impl_associated_item_for_impl_trait_in_trait(
    tcx: TyCtxt<'_>,
    trait_assoc_def_id: LocalDefId,
    impl_fn_def_id: LocalDefId,
) -> LocalDefId {
    let impl_def_id = tcx.local_parent(impl_fn_def_id);

    let span = tcx.def_span(trait_assoc_def_id);
    let impl_assoc_ty = tcx.at(span).create_def(impl_def_id, DefPathData::ImplTraitAssocTy);

    impl_assoc_ty.def_id()
}
