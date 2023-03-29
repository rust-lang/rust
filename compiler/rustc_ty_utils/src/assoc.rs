use rustc_data_structures::fx::FxIndexSet;
use rustc_hir as hir;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{DefId, DefIdMap, LocalDefId};
use rustc_hir::definitions::DefPathData;
use rustc_hir::intravisit::{self, Visitor};
use rustc_middle::ty::{self, ImplTraitInTraitData, InternalSubsts, TyCtxt};
use rustc_span::symbol::kw;

pub fn provide(providers: &mut ty::query::Providers) {
    *providers = ty::query::Providers {
        associated_item,
        associated_item_def_ids,
        associated_items,
        associated_types_for_impl_traits_in_associated_fn,
        associated_type_for_impl_trait_in_trait,
        impl_item_implementor_ids,
        ..*providers
    };
}

fn associated_item_def_ids(tcx: TyCtxt<'_>, def_id: LocalDefId) -> &[DefId] {
    let item = tcx.hir().expect_item(def_id);
    match item.kind {
        hir::ItemKind::Trait(.., ref trait_item_refs) => {
            if tcx.lower_impl_trait_in_trait_to_assoc_ty() {
                // We collect RPITITs for each trait method's return type and create a
                // corresponding associated item using associated_types_for_impl_traits_in_associated_fn
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
                                    tcx.associated_types_for_impl_traits_in_associated_fn(
                                        trait_fn_def_id,
                                    )
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
        hir::ItemKind::Impl(ref impl_) => {
            if tcx.lower_impl_trait_in_trait_to_assoc_ty() {
                // We collect RPITITs for each trait method's return type, on the impl side too and
                // create a corresponding associated item using
                // associated_types_for_impl_traits_in_associated_fn query.
                tcx.arena.alloc_from_iter(
                    impl_
                        .items
                        .iter()
                        .map(|impl_item_ref| impl_item_ref.id.owner_id.to_def_id())
                        .chain(impl_.of_trait.iter().flat_map(|_| {
                            impl_
                                .items
                                .iter()
                                .filter(|impl_item_ref| {
                                    matches!(impl_item_ref.kind, hir::AssocItemKind::Fn { .. })
                                })
                                .flat_map(|impl_item_ref| {
                                    let impl_fn_def_id =
                                        impl_item_ref.id.owner_id.def_id.to_def_id();
                                    tcx.associated_types_for_impl_traits_in_associated_fn(
                                        impl_fn_def_id,
                                    )
                                })
                                .map(|def_id| *def_id)
                        })),
                )
            } else {
                tcx.arena.alloc_from_iter(
                    impl_.items.iter().map(|impl_item_ref| impl_item_ref.id.owner_id.to_def_id()),
                )
            }
        }
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

fn impl_item_implementor_ids(tcx: TyCtxt<'_>, impl_id: DefId) -> DefIdMap<DefId> {
    tcx.associated_items(impl_id)
        .in_definition_order()
        .filter_map(|item| item.trait_item_def_id.map(|trait_item| (trait_item, item.def_id)))
        .collect()
}

fn associated_item(tcx: TyCtxt<'_>, def_id: LocalDefId) -> ty::AssocItem {
    let id = tcx.hir().local_def_id_to_hir_id(def_id);
    let parent_def_id = tcx.hir().get_parent_item(id);
    let parent_item = tcx.hir().expect_item(parent_def_id.def_id);
    match parent_item.kind {
        hir::ItemKind::Impl(ref impl_) => {
            if let Some(impl_item_ref) = impl_.items.iter().find(|i| i.id.owner_id.def_id == def_id)
            {
                let assoc_item = associated_item_from_impl_item_ref(impl_item_ref);
                debug_assert_eq!(assoc_item.def_id.expect_local(), def_id);
                return assoc_item;
            }
        }

        hir::ItemKind::Trait(.., ref trait_item_refs) => {
            if let Some(trait_item_ref) =
                trait_item_refs.iter().find(|i| i.id.owner_id.def_id == def_id)
            {
                let assoc_item = associated_item_from_trait_item_ref(trait_item_ref);
                debug_assert_eq!(assoc_item.def_id.expect_local(), def_id);
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
        opt_rpitit_info: None,
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
        opt_rpitit_info: None,
    }
}

/// Given an `fn_def_id` of a trait or a trait implementation:
///
/// if `fn_def_id` is a function defined inside a trait, then it synthesizes
/// a new def id corresponding to a new associated type for each return-
/// position `impl Trait` in the signature.
///
/// if `fn_def_id` is a function inside of an impl, then for each synthetic
/// associated type generated for the corresponding trait function described
/// above, synthesize a corresponding associated type in the impl.
fn associated_types_for_impl_traits_in_associated_fn(
    tcx: TyCtxt<'_>,
    fn_def_id: LocalDefId,
) -> &'_ [DefId] {
    let parent_def_id = tcx.local_parent(fn_def_id);

    match tcx.def_kind(parent_def_id) {
        DefKind::Trait => {
            struct RPITVisitor<'tcx> {
                rpits: FxIndexSet<LocalDefId>,
                tcx: TyCtxt<'tcx>,
            }

            impl<'tcx> Visitor<'tcx> for RPITVisitor<'tcx> {
                fn visit_ty(&mut self, ty: &'tcx hir::Ty<'tcx>) {
                    if let hir::TyKind::OpaqueDef(item_id, _, _) = ty.kind
                        && self.rpits.insert(item_id.owner_id.def_id)
                    {
                        let opaque_item = self.tcx.hir().expect_item(item_id.owner_id.def_id).expect_opaque_ty();
                        for bound in opaque_item.bounds {
                            intravisit::walk_param_bound(self, bound);
                        }
                    }
                    intravisit::walk_ty(self, ty)
                }
            }

            let mut visitor = RPITVisitor { tcx, rpits: FxIndexSet::default() };

            if let Some(output) = tcx.hir().get_fn_output(fn_def_id) {
                visitor.visit_fn_ret_ty(output);

                tcx.arena.alloc_from_iter(visitor.rpits.iter().map(|opaque_ty_def_id| {
                    tcx.associated_type_for_impl_trait_in_trait(opaque_ty_def_id).to_def_id()
                }))
            } else {
                &[]
            }
        }

        DefKind::Impl { .. } => {
            let Some(trait_fn_def_id) = tcx.associated_item(fn_def_id).trait_item_def_id else { return &[] };

            tcx.arena.alloc_from_iter(
                tcx.associated_types_for_impl_traits_in_associated_fn(trait_fn_def_id).iter().map(
                    move |&trait_assoc_def_id| {
                        associated_type_for_impl_trait_in_impl(tcx, trait_assoc_def_id, fn_def_id)
                            .to_def_id()
                    },
                ),
            )
        }

        def_kind => bug!(
            "associated_types_for_impl_traits_in_associated_fn: {:?} should be Trait or Impl but is {:?}",
            parent_def_id,
            def_kind
        ),
    }
}

/// Given an `opaque_ty_def_id` corresponding to an `impl Trait` in an associated
/// function from a trait, synthesize an associated type for that `impl Trait`
/// that inherits properties that we infer from the method and the opaque type.
fn associated_type_for_impl_trait_in_trait(
    tcx: TyCtxt<'_>,
    opaque_ty_def_id: LocalDefId,
) -> LocalDefId {
    let (hir::OpaqueTyOrigin::FnReturn(fn_def_id) | hir::OpaqueTyOrigin::AsyncFn(fn_def_id)) =
        tcx.hir().expect_item(opaque_ty_def_id).expect_opaque_ty().origin
    else {
        bug!("expected opaque for {opaque_ty_def_id:?}");
    };
    let trait_def_id = tcx.local_parent(fn_def_id);
    assert_eq!(tcx.def_kind(trait_def_id), DefKind::Trait);

    let span = tcx.def_span(opaque_ty_def_id);
    let trait_assoc_ty = tcx.at(span).create_def(trait_def_id, DefPathData::ImplTraitAssocTy);

    let local_def_id = trait_assoc_ty.def_id();
    let def_id = local_def_id.to_def_id();

    trait_assoc_ty.opt_def_kind(Some(DefKind::AssocTy));

    // There's no HIR associated with this new synthesized `def_id`, so feed
    // `opt_local_def_id_to_hir_id` with `None`.
    trait_assoc_ty.opt_local_def_id_to_hir_id(None);

    // Copy span of the opaque.
    trait_assoc_ty.def_ident_span(Some(span));

    trait_assoc_ty.associated_item(ty::AssocItem {
        name: kw::Empty,
        kind: ty::AssocKind::Type,
        def_id,
        trait_item_def_id: None,
        container: ty::TraitContainer,
        fn_has_self_parameter: false,
        opt_rpitit_info: Some(ImplTraitInTraitData::Trait {
            fn_def_id: fn_def_id.to_def_id(),
            opaque_def_id: opaque_ty_def_id.to_def_id(),
        }),
    });

    // Copy visility of the containing function.
    trait_assoc_ty.visibility(tcx.visibility(fn_def_id));

    // Copy impl_defaultness of the containing function.
    trait_assoc_ty.impl_defaultness(tcx.impl_defaultness(fn_def_id));

    // Copy type_of of the opaque.
    trait_assoc_ty.type_of(ty::EarlyBinder(tcx.mk_opaque(
        opaque_ty_def_id.to_def_id(),
        InternalSubsts::identity_for_item(tcx, opaque_ty_def_id),
    )));

    trait_assoc_ty.is_type_alias_impl_trait(false);

    // Copy generics_of of the opaque type item but the trait is the parent.
    trait_assoc_ty.generics_of({
        let opaque_ty_generics = tcx.generics_of(opaque_ty_def_id);
        let opaque_ty_parent_count = opaque_ty_generics.parent_count;
        let mut params = opaque_ty_generics.params.clone();

        let parent_generics = tcx.generics_of(trait_def_id);
        let parent_count = parent_generics.parent_count + parent_generics.params.len();

        let mut trait_fn_params = tcx.generics_of(fn_def_id).params.clone();

        for param in &mut params {
            param.index = param.index + parent_count as u32 + trait_fn_params.len() as u32
                - opaque_ty_parent_count as u32;
        }

        trait_fn_params.extend(params);
        params = trait_fn_params;

        let param_def_id_to_index =
            params.iter().map(|param| (param.def_id, param.index)).collect();

        ty::Generics {
            parent: Some(trait_def_id.to_def_id()),
            parent_count,
            params,
            param_def_id_to_index,
            has_self: false,
            has_late_bound_regions: opaque_ty_generics.has_late_bound_regions,
            defines_opaque_types: vec![],
        }
    });

    // There are no predicates for the synthesized associated type.
    trait_assoc_ty.explicit_predicates_of(ty::GenericPredicates {
        parent: Some(trait_def_id.to_def_id()),
        predicates: &[],
    });

    // There are no inferred outlives for the synthesized associated type.
    trait_assoc_ty.inferred_outlives_of(&[]);

    local_def_id
}

/// Given an `trait_assoc_def_id` corresponding to an associated item synthesized
/// from an `impl Trait` in an associated function from a trait, and an
/// `impl_fn_def_id` that represents an implementation of the associated function
/// that the `impl Trait` comes from, synthesize an associated type for that `impl Trait`
/// that inherits properties that we infer from the method and the associated type.
fn associated_type_for_impl_trait_in_impl(
    tcx: TyCtxt<'_>,
    trait_assoc_def_id: DefId,
    impl_fn_def_id: LocalDefId,
) -> LocalDefId {
    let impl_local_def_id = tcx.local_parent(impl_fn_def_id);

    // FIXME fix the span, we probably want the def_id of the return type of the function
    let span = tcx.def_span(impl_fn_def_id);
    let impl_assoc_ty = tcx.at(span).create_def(impl_local_def_id, DefPathData::ImplTraitAssocTy);

    let local_def_id = impl_assoc_ty.def_id();
    let def_id = local_def_id.to_def_id();

    impl_assoc_ty.opt_def_kind(Some(DefKind::AssocTy));

    // There's no HIR associated with this new synthesized `def_id`, so feed
    // `opt_local_def_id_to_hir_id` with `None`.
    impl_assoc_ty.opt_local_def_id_to_hir_id(None);

    // Copy span of the opaque.
    impl_assoc_ty.def_ident_span(Some(span));

    impl_assoc_ty.associated_item(ty::AssocItem {
        name: kw::Empty,
        kind: ty::AssocKind::Type,
        def_id,
        trait_item_def_id: Some(trait_assoc_def_id),
        container: ty::ImplContainer,
        fn_has_self_parameter: false,
        opt_rpitit_info: Some(ImplTraitInTraitData::Impl { fn_def_id: impl_fn_def_id.to_def_id() }),
    });

    // Copy visility of the containing function.
    impl_assoc_ty.visibility(tcx.visibility(impl_fn_def_id));

    // Copy impl_defaultness of the containing function.
    impl_assoc_ty.impl_defaultness(tcx.impl_defaultness(impl_fn_def_id));

    // Copy generics_of the trait's associated item but the impl as the parent.
    // FIXME(-Zlower-impl-trait-in-trait-to-assoc-ty) resolves to the trait instead of the impl
    // generics.
    impl_assoc_ty.generics_of({
        let trait_assoc_generics = tcx.generics_of(trait_assoc_def_id);
        let trait_assoc_parent_count = trait_assoc_generics.parent_count;
        let mut params = trait_assoc_generics.params.clone();

        let parent_generics = tcx.generics_of(impl_local_def_id.to_def_id());
        let parent_count = parent_generics.parent_count + parent_generics.params.len();

        for param in &mut params {
            param.index = param.index + parent_count as u32 - trait_assoc_parent_count as u32;
        }

        let param_def_id_to_index =
            params.iter().map(|param| (param.def_id, param.index)).collect();

        ty::Generics {
            parent: Some(impl_local_def_id.to_def_id()),
            parent_count,
            params,
            param_def_id_to_index,
            has_self: false,
            has_late_bound_regions: trait_assoc_generics.has_late_bound_regions,
            defines_opaque_types: vec![],
        }
    });

    // There are no predicates for the synthesized associated type.
    impl_assoc_ty.explicit_predicates_of(ty::GenericPredicates {
        parent: Some(impl_local_def_id.to_def_id()),
        predicates: &[],
    });

    // There are no inferred outlives for the synthesized associated type.
    impl_assoc_ty.inferred_outlives_of(&[]);

    local_def_id
}
