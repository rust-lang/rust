use rustc_data_structures::fx::FxIndexSet;
use rustc_hir as hir;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{DefId, DefIdMap, LocalDefId};
use rustc_hir::intravisit::{self, Visitor};
use rustc_middle::query::Providers;
use rustc_middle::ty::{self, ImplTraitInTraitData, Ty, TyCtxt};
use rustc_middle::{bug, span_bug};
use rustc_span::sym;
use rustc_span::symbol::kw;

pub(crate) fn provide(providers: &mut Providers) {
    *providers = Providers {
        associated_item,
        associated_item_def_ids,
        associated_items,
        associated_types_for_impl_traits_in_associated_fn,
        associated_type_for_effects,
        associated_type_for_impl_trait_in_trait,
        impl_item_implementor_ids,
        ..*providers
    };
}

fn associated_item_def_ids(tcx: TyCtxt<'_>, def_id: LocalDefId) -> &[DefId] {
    let item = tcx.hir().expect_item(def_id);
    match item.kind {
        hir::ItemKind::Trait(.., trait_item_refs) => {
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
                                let trait_fn_def_id = trait_item_ref.id.owner_id.def_id.to_def_id();
                                tcx.associated_types_for_impl_traits_in_associated_fn(
                                    trait_fn_def_id,
                                )
                            })
                            .copied(),
                    )
                    .chain(tcx.associated_type_for_effects(def_id)),
            )
        }
        hir::ItemKind::Impl(impl_) => {
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
                                let impl_fn_def_id = impl_item_ref.id.owner_id.def_id.to_def_id();
                                tcx.associated_types_for_impl_traits_in_associated_fn(
                                    impl_fn_def_id,
                                )
                            })
                            .copied()
                    }))
                    .chain(tcx.associated_type_for_effects(def_id)),
            )
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
    let id = tcx.local_def_id_to_hir_id(def_id);
    let parent_def_id = tcx.hir().get_parent_item(id);
    let parent_item = tcx.hir().expect_item(parent_def_id.def_id);
    match parent_item.kind {
        hir::ItemKind::Impl(impl_) => {
            if let Some(impl_item_ref) = impl_.items.iter().find(|i| i.id.owner_id.def_id == def_id)
            {
                let assoc_item = associated_item_from_impl_item_ref(impl_item_ref);
                debug_assert_eq!(assoc_item.def_id.expect_local(), def_id);
                return assoc_item;
            }
        }

        hir::ItemKind::Trait(.., trait_item_refs) => {
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
        is_effects_desugaring: false,
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
        is_effects_desugaring: false,
    }
}

/// Given an `def_id` of a trait or a trait impl:
///
/// If `def_id` is a trait that has `#[const_trait]`, then it synthesizes
/// a new def id corresponding to a new associated type for the effects.
///
/// If `def_id` is an impl, then synthesize the associated type according
/// to the constness of the impl.
fn associated_type_for_effects(tcx: TyCtxt<'_>, def_id: LocalDefId) -> Option<DefId> {
    // don't synthesize the associated type even if the user has written `const_trait`
    // if the effects feature is disabled.
    if !tcx.features().effects {
        return None;
    }
    let (feed, parent_did) = match tcx.def_kind(def_id) {
        DefKind::Trait => {
            let trait_def_id = def_id;
            let attr = tcx.get_attr(def_id, sym::const_trait)?;

            let span = attr.span;
            let trait_assoc_ty = tcx.at(span).create_def(trait_def_id, kw::Empty, DefKind::AssocTy);

            let local_def_id = trait_assoc_ty.def_id();
            let def_id = local_def_id.to_def_id();

            // Copy span of the attribute.
            trait_assoc_ty.def_ident_span(Some(span));

            trait_assoc_ty.associated_item(ty::AssocItem {
                name: kw::Empty,
                kind: ty::AssocKind::Type,
                def_id,
                trait_item_def_id: None,
                container: ty::TraitContainer,
                fn_has_self_parameter: false,
                opt_rpitit_info: None,
                is_effects_desugaring: true,
            });

            // No default type
            trait_assoc_ty.defaultness(hir::Defaultness::Default { has_value: false });

            trait_assoc_ty.is_type_alias_impl_trait(false);

            (trait_assoc_ty, trait_def_id)
        }
        DefKind::Impl { .. } => {
            let impl_def_id = def_id;
            let trait_id = tcx.trait_id_of_impl(def_id.to_def_id())?;

            // first get the DefId of the assoc type on the trait, if there is not,
            // then we don't need to generate it on the impl.
            let trait_assoc_id = tcx.associated_type_for_effects(trait_id)?;

            // FIXME(effects): span
            let span = tcx.def_ident_span(def_id).unwrap();

            let impl_assoc_ty = tcx.at(span).create_def(def_id, kw::Empty, DefKind::AssocTy);

            let local_def_id = impl_assoc_ty.def_id();
            let def_id = local_def_id.to_def_id();

            impl_assoc_ty.def_ident_span(Some(span));

            impl_assoc_ty.associated_item(ty::AssocItem {
                name: kw::Empty,
                kind: ty::AssocKind::Type,
                def_id,
                trait_item_def_id: Some(trait_assoc_id),
                container: ty::ImplContainer,
                fn_has_self_parameter: false,
                opt_rpitit_info: None,
                is_effects_desugaring: true,
            });

            // no default value.
            impl_assoc_ty.defaultness(hir::Defaultness::Final);

            // set the type of the associated type! If this is a const impl,
            // we set to Maybe, otherwise we set to `Runtime`.
            let type_def_id = if tcx.is_const_trait_impl_raw(impl_def_id.to_def_id()) {
                tcx.require_lang_item(hir::LangItem::EffectsMaybe, Some(span))
            } else {
                tcx.require_lang_item(hir::LangItem::EffectsRuntime, Some(span))
            };
            // FIXME(effects): make impls use `Min` for their effect types
            impl_assoc_ty.type_of(ty::EarlyBinder::bind(Ty::new_adt(
                tcx,
                tcx.adt_def(type_def_id),
                ty::GenericArgs::empty(),
            )));

            (impl_assoc_ty, impl_def_id)
        }
        def_kind => bug!(
            "associated_type_for_effects: {:?} should be Trait or Impl but is {:?}",
            def_id,
            def_kind
        ),
    };

    feed.feed_hir();

    // visibility is public.
    feed.visibility(ty::Visibility::Public);

    // Copy generics_of of the trait/impl, making the trait/impl as parent.
    feed.generics_of({
        let parent_generics = tcx.generics_of(parent_did);
        let parent_count = parent_generics.parent_count + parent_generics.own_params.len();

        ty::Generics {
            parent: Some(parent_did.to_def_id()),
            parent_count,
            own_params: vec![],
            param_def_id_to_index: parent_generics.param_def_id_to_index.clone(),
            has_self: false,
            has_late_bound_regions: None,
            host_effect_index: parent_generics.host_effect_index,
        }
    });
    feed.explicit_item_super_predicates(ty::EarlyBinder::bind(&[]));

    // There are no inferred outlives for the synthesized associated type.
    feed.inferred_outlives_of(&[]);

    Some(feed.def_id().to_def_id())
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
            struct RPITVisitor {
                rpits: FxIndexSet<LocalDefId>,
            }

            impl<'tcx> Visitor<'tcx> for RPITVisitor {
                fn visit_ty(&mut self, ty: &'tcx hir::Ty<'tcx>) {
                    if let hir::TyKind::OpaqueDef(opaq, _) = ty.kind
                        && self.rpits.insert(opaq.def_id)
                    {
                        for bound in opaq.bounds {
                            intravisit::walk_param_bound(self, bound);
                        }
                    }
                    intravisit::walk_ty(self, ty)
                }
            }

            let mut visitor = RPITVisitor { rpits: FxIndexSet::default() };

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
            let Some(trait_fn_def_id) = tcx.associated_item(fn_def_id).trait_item_def_id else {
                return &[];
            };

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
    let (hir::OpaqueTyOrigin::FnReturn { parent: fn_def_id, .. }
    | hir::OpaqueTyOrigin::AsyncFn { parent: fn_def_id, .. }) =
        tcx.opaque_type_origin(opaque_ty_def_id)
    else {
        bug!("expected opaque for {opaque_ty_def_id:?}");
    };
    let trait_def_id = tcx.local_parent(fn_def_id);
    assert_eq!(tcx.def_kind(trait_def_id), DefKind::Trait);

    let span = tcx.def_span(opaque_ty_def_id);
    let trait_assoc_ty = tcx.at(span).create_def(trait_def_id, kw::Empty, DefKind::AssocTy);

    let local_def_id = trait_assoc_ty.def_id();
    let def_id = local_def_id.to_def_id();

    trait_assoc_ty.feed_hir();

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
        is_effects_desugaring: false,
    });

    // Copy visility of the containing function.
    trait_assoc_ty.visibility(tcx.visibility(fn_def_id));

    // Copy defaultness of the containing function.
    trait_assoc_ty.defaultness(tcx.defaultness(fn_def_id));

    trait_assoc_ty.is_type_alias_impl_trait(false);

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

    let decl = tcx.hir_node_by_def_id(impl_fn_def_id).fn_decl().expect("expected decl");
    let span = match decl.output {
        hir::FnRetTy::DefaultReturn(_) => tcx.def_span(impl_fn_def_id),
        hir::FnRetTy::Return(ty) => ty.span,
    };
    let impl_assoc_ty = tcx.at(span).create_def(impl_local_def_id, kw::Empty, DefKind::AssocTy);

    let local_def_id = impl_assoc_ty.def_id();
    let def_id = local_def_id.to_def_id();

    impl_assoc_ty.feed_hir();

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
        is_effects_desugaring: false,
    });

    // Copy visility of the containing function.
    impl_assoc_ty.visibility(tcx.visibility(impl_fn_def_id));

    // Copy defaultness of the containing function.
    impl_assoc_ty.defaultness(tcx.defaultness(impl_fn_def_id));

    // Copy generics_of the trait's associated item but the impl as the parent.
    // FIXME: This may be detrimental to diagnostics, as we resolve the early-bound vars
    // here to paramswhose parent are items in the trait. We could synthesize new params
    // here, but it seems overkill.
    impl_assoc_ty.generics_of({
        let trait_assoc_generics = tcx.generics_of(trait_assoc_def_id);
        let trait_assoc_parent_count = trait_assoc_generics.parent_count;
        let mut own_params = trait_assoc_generics.own_params.clone();

        let parent_generics = tcx.generics_of(impl_local_def_id.to_def_id());
        let parent_count = parent_generics.parent_count + parent_generics.own_params.len();

        for param in &mut own_params {
            param.index = param.index + parent_count as u32 - trait_assoc_parent_count as u32;
        }

        let param_def_id_to_index =
            own_params.iter().map(|param| (param.def_id, param.index)).collect();

        ty::Generics {
            parent: Some(impl_local_def_id.to_def_id()),
            parent_count,
            own_params,
            param_def_id_to_index,
            has_self: false,
            has_late_bound_regions: trait_assoc_generics.has_late_bound_regions,
            host_effect_index: parent_generics.host_effect_index,
        }
    });

    // There are no inferred outlives for the synthesized associated type.
    impl_assoc_ty.inferred_outlives_of(&[]);

    local_def_id
}
