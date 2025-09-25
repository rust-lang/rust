use rustc_hir::def::DefKind;
use rustc_hir::def_id::{DefId, DefIdMap, LocalDefId};
use rustc_hir::definitions::{DefPathData, DisambiguatorState};
use rustc_hir::intravisit::{self, Visitor};
use rustc_hir::{self as hir, ImplItemImplKind, ItemKind};
use rustc_middle::query::Providers;
use rustc_middle::ty::{self, ImplTraitInTraitData, TyCtxt};
use rustc_middle::{bug, span_bug};
use rustc_span::Ident;
use rustc_span::symbol::kw;

pub(crate) fn provide(providers: &mut Providers) {
    *providers = Providers {
        associated_item,
        associated_item_def_ids,
        associated_items,
        associated_types_for_impl_traits_in_trait_or_impl,
        impl_item_implementor_ids,
        ..*providers
    };
}

fn associated_item_def_ids(tcx: TyCtxt<'_>, def_id: LocalDefId) -> &[DefId] {
    let item = tcx.hir_expect_item(def_id);
    match item.kind {
        hir::ItemKind::Trait(.., trait_item_refs) => {
            // We collect RPITITs for each trait method's return type and create a corresponding
            // associated item using the associated_types_for_impl_traits_in_trait_or_impl
            // query.
            let rpitit_items = tcx.associated_types_for_impl_traits_in_trait_or_impl(def_id);
            tcx.arena.alloc_from_iter(trait_item_refs.iter().flat_map(|trait_item_ref| {
                let item_def_id = trait_item_ref.owner_id.to_def_id();
                [item_def_id]
                    .into_iter()
                    .chain(rpitit_items.get(&item_def_id).into_iter().flatten().copied())
            }))
        }
        hir::ItemKind::Impl(impl_) => {
            // We collect RPITITs for each trait method's return type, on the impl side too and
            // create a corresponding associated item using
            // associated_types_for_impl_traits_in_trait_or_impl query.
            let rpitit_items = tcx.associated_types_for_impl_traits_in_trait_or_impl(def_id);
            tcx.arena.alloc_from_iter(impl_.items.iter().flat_map(|impl_item_ref| {
                let item_def_id = impl_item_ref.owner_id.to_def_id();
                [item_def_id]
                    .into_iter()
                    .chain(rpitit_items.get(&item_def_id).into_iter().flatten().copied())
            }))
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
        .filter_map(|item| item.trait_item_def_id().map(|trait_item| (trait_item, item.def_id)))
        .collect()
}

fn associated_item(tcx: TyCtxt<'_>, def_id: LocalDefId) -> ty::AssocItem {
    let assoc_item = match tcx.hir_node_by_def_id(def_id) {
        hir::Node::TraitItem(ti) => associated_item_from_trait_item(tcx, ti),
        hir::Node::ImplItem(ii) => associated_item_from_impl_item(tcx, ii),
        node => span_bug!(tcx.def_span(def_id), "impl item or item not found: {:?}", node,),
    };
    debug_assert_eq!(assoc_item.def_id.expect_local(), def_id);
    assoc_item
}

fn fn_has_self_parameter(tcx: TyCtxt<'_>, owner_id: hir::OwnerId) -> bool {
    matches!(tcx.fn_arg_idents(owner_id.def_id), [Some(Ident { name: kw::SelfLower, .. }), ..])
}

fn associated_item_from_trait_item(
    tcx: TyCtxt<'_>,
    trait_item: &hir::TraitItem<'_>,
) -> ty::AssocItem {
    let owner_id = trait_item.owner_id;
    let name = trait_item.ident.name;
    let kind = match trait_item.kind {
        hir::TraitItemKind::Const { .. } => ty::AssocKind::Const { name },
        hir::TraitItemKind::Fn { .. } => {
            ty::AssocKind::Fn { name, has_self: fn_has_self_parameter(tcx, owner_id) }
        }
        hir::TraitItemKind::Type { .. } => {
            ty::AssocKind::Type { data: ty::AssocTypeData::Normal(name) }
        }
    };

    ty::AssocItem { kind, def_id: owner_id.to_def_id(), container: ty::AssocContainer::Trait }
}

fn associated_item_from_impl_item(tcx: TyCtxt<'_>, impl_item: &hir::ImplItem<'_>) -> ty::AssocItem {
    let owner_id = impl_item.owner_id;
    let name = impl_item.ident.name;
    let kind = match impl_item.kind {
        hir::ImplItemKind::Const { .. } => ty::AssocKind::Const { name },
        hir::ImplItemKind::Fn { .. } => {
            ty::AssocKind::Fn { name, has_self: fn_has_self_parameter(tcx, owner_id) }
        }
        hir::ImplItemKind::Type { .. } => {
            ty::AssocKind::Type { data: ty::AssocTypeData::Normal(name) }
        }
    };

    let container = match impl_item.impl_kind {
        ImplItemImplKind::Inherent { .. } => ty::AssocContainer::InherentImpl,
        ImplItemImplKind::Trait { trait_item_def_id, .. } => {
            ty::AssocContainer::TraitImpl(trait_item_def_id)
        }
    };
    ty::AssocItem { kind, def_id: owner_id.to_def_id(), container }
}
struct RPITVisitor<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    synthetics: Vec<LocalDefId>,
    data: DefPathData,
    disambiguator: &'a mut DisambiguatorState,
}

impl<'tcx> Visitor<'tcx> for RPITVisitor<'_, 'tcx> {
    fn visit_opaque_ty(&mut self, opaque: &'tcx hir::OpaqueTy<'tcx>) -> Self::Result {
        self.synthetics.push(associated_type_for_impl_trait_in_trait(
            self.tcx,
            opaque.def_id,
            self.data,
            &mut self.disambiguator,
        ));
        intravisit::walk_opaque_ty(self, opaque)
    }
}

fn associated_types_for_impl_traits_in_trait_or_impl<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: LocalDefId,
) -> DefIdMap<Vec<DefId>> {
    let item = tcx.hir_expect_item(def_id);
    let disambiguator = &mut DisambiguatorState::new();
    match item.kind {
        ItemKind::Trait(.., trait_item_refs) => trait_item_refs
            .iter()
            .filter_map(move |item| {
                if !matches!(tcx.def_kind(item.owner_id), DefKind::AssocFn) {
                    return None;
                }
                let fn_def_id = item.owner_id.def_id;
                let Some(output) = tcx.hir_get_fn_output(fn_def_id) else {
                    return Some((fn_def_id.to_def_id(), vec![]));
                };
                let def_name = tcx.item_name(fn_def_id.to_def_id());
                let data = DefPathData::AnonAssocTy(def_name);
                let mut visitor = RPITVisitor { tcx, synthetics: vec![], data, disambiguator };
                visitor.visit_fn_ret_ty(output);
                let defs = visitor
                    .synthetics
                    .into_iter()
                    .map(|def_id| def_id.to_def_id())
                    .collect::<Vec<_>>();
                Some((fn_def_id.to_def_id(), defs))
            })
            .collect(),
        ItemKind::Impl(impl_) => {
            let Some(of_trait) = impl_.of_trait else {
                return Default::default();
            };
            let Some(trait_def_id) = of_trait.trait_ref.trait_def_id() else {
                return Default::default();
            };
            let in_trait_def = tcx.associated_types_for_impl_traits_in_trait_or_impl(trait_def_id);
            impl_
                .items
                .iter()
                .filter_map(|item| {
                    if !matches!(tcx.def_kind(item.owner_id), DefKind::AssocFn) {
                        return None;
                    }
                    let did = item.owner_id.def_id.to_def_id();
                    let item = tcx.hir_impl_item(*item);
                    let ImplItemImplKind::Trait {
                        trait_item_def_id: Ok(trait_item_def_id), ..
                    } = item.impl_kind
                    else {
                        return Some((did, vec![]));
                    };
                    let iter = in_trait_def[&trait_item_def_id].iter().map(|&id| {
                        associated_type_for_impl_trait_in_impl(tcx, id, item, disambiguator)
                            .to_def_id()
                    });
                    Some((did, iter.collect()))
                })
                .collect()
        }
        _ => {
            bug!(
                "associated_types_for_impl_traits_in_trait_or_impl: {:?} should be Trait or Impl but is {:?}",
                def_id,
                tcx.def_kind(def_id)
            )
        }
    }
}

/// Given an `opaque_ty_def_id` corresponding to an `impl Trait` in an associated
/// function from a trait, synthesize an associated type for that `impl Trait`
/// that inherits properties that we infer from the method and the opaque type.
fn associated_type_for_impl_trait_in_trait(
    tcx: TyCtxt<'_>,
    opaque_ty_def_id: LocalDefId,
    data: DefPathData,
    disambiguator: &mut DisambiguatorState,
) -> LocalDefId {
    let (hir::OpaqueTyOrigin::FnReturn { parent: fn_def_id, .. }
    | hir::OpaqueTyOrigin::AsyncFn { parent: fn_def_id, .. }) =
        tcx.local_opaque_ty_origin(opaque_ty_def_id)
    else {
        bug!("expected opaque for {opaque_ty_def_id:?}");
    };
    let trait_def_id = tcx.local_parent(fn_def_id);
    assert_eq!(tcx.def_kind(trait_def_id), DefKind::Trait);

    let span = tcx.def_span(opaque_ty_def_id);
    // Also use the method name to create an unique def path.
    let trait_assoc_ty = tcx.at(span).create_def(
        trait_def_id,
        // No name because this is an anonymous associated type.
        None,
        DefKind::AssocTy,
        Some(data),
        disambiguator,
    );

    let local_def_id = trait_assoc_ty.def_id();
    let def_id = local_def_id.to_def_id();

    trait_assoc_ty.feed_hir();

    // Copy span of the opaque.
    trait_assoc_ty.def_ident_span(Some(span));

    trait_assoc_ty.associated_item(ty::AssocItem {
        kind: ty::AssocKind::Type {
            data: ty::AssocTypeData::Rpitit(ImplTraitInTraitData::Trait {
                fn_def_id: fn_def_id.to_def_id(),
                opaque_def_id: opaque_ty_def_id.to_def_id(),
            }),
        },
        def_id,
        container: ty::AssocContainer::Trait,
    });

    // Copy visility of the containing function.
    trait_assoc_ty.visibility(tcx.visibility(fn_def_id));

    // Copy defaultness of the containing function.
    trait_assoc_ty.defaultness(tcx.defaultness(fn_def_id));

    // There are no inferred outlives for the synthesized associated type.
    trait_assoc_ty.inferred_outlives_of(&[]);

    local_def_id
}

/// Given an `trait_assoc_def_id` corresponding to an associated item synthesized
/// from an `impl Trait` in an associated function from a trait, and an
/// `impl_fn` that represents an implementation of the associated function
/// that the `impl Trait` comes from, synthesize an associated type for that `impl Trait`
/// that inherits properties that we infer from the method and the associated type.
fn associated_type_for_impl_trait_in_impl(
    tcx: TyCtxt<'_>,
    trait_assoc_def_id: DefId,
    impl_fn: &hir::ImplItem<'_>,
    disambiguator: &mut DisambiguatorState,
) -> LocalDefId {
    let impl_local_def_id = tcx.local_parent(impl_fn.owner_id.def_id);

    let hir::ImplItemKind::Fn(fn_sig, _) = impl_fn.kind else { bug!("expected decl") };
    let span = match fn_sig.decl.output {
        hir::FnRetTy::DefaultReturn(_) => tcx.def_span(impl_fn.owner_id),
        hir::FnRetTy::Return(ty) => ty.span,
    };

    // Use the same disambiguator and method name as the anon associated type in the trait.
    let disambiguated_data = tcx.def_key(trait_assoc_def_id).disambiguated_data;
    let DefPathData::AnonAssocTy(name) = disambiguated_data.data else {
        bug!("expected anon associated type")
    };
    let data = DefPathData::AnonAssocTy(name);

    let impl_assoc_ty = tcx.at(span).create_def(
        impl_local_def_id,
        // No name because this is an anonymous associated type.
        None,
        DefKind::AssocTy,
        Some(data),
        disambiguator,
    );

    let local_def_id = impl_assoc_ty.def_id();
    let def_id = local_def_id.to_def_id();

    impl_assoc_ty.feed_hir();

    // Copy span of the opaque.
    impl_assoc_ty.def_ident_span(Some(span));

    impl_assoc_ty.associated_item(ty::AssocItem {
        kind: ty::AssocKind::Type {
            data: ty::AssocTypeData::Rpitit(ImplTraitInTraitData::Impl {
                fn_def_id: impl_fn.owner_id.to_def_id(),
            }),
        },
        def_id,
        container: ty::AssocContainer::TraitImpl(Ok(trait_assoc_def_id)),
    });

    // Copy visility of the containing function.
    impl_assoc_ty.visibility(tcx.visibility(impl_fn.owner_id));

    // Copy defaultness of the containing function.
    impl_assoc_ty.defaultness(tcx.defaultness(impl_fn.owner_id));

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
        }
    });

    // There are no inferred outlives for the synthesized associated type.
    impl_assoc_ty.inferred_outlives_of(&[]);

    local_def_id
}
