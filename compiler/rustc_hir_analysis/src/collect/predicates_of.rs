use std::assert_matches::assert_matches;

use hir::{HirId, Node};
use rustc_data_structures::fx::FxIndexSet;
use rustc_hir as hir;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::intravisit::{self, Visitor};
use rustc_middle::ty::{self, GenericPredicates, ImplTraitInTraitData, Ty, TyCtxt, Upcast};
use rustc_middle::{bug, span_bug};
use rustc_span::symbol::Ident;
use rustc_span::{DUMMY_SP, Span};
use tracing::{debug, instrument, trace};

use crate::bounds::Bounds;
use crate::collect::ItemCtxt;
use crate::constrained_generic_params as cgp;
use crate::delegation::inherit_predicates_for_delegation_item;
use crate::hir_ty_lowering::{HirTyLowerer, PredicateFilter, RegionInferReason};

/// Returns a list of all type predicates (explicit and implicit) for the definition with
/// ID `def_id`. This includes all predicates returned by `explicit_predicates_of`, plus
/// inferred constraints concerning which regions outlive other regions.
#[instrument(level = "debug", skip(tcx))]
pub(super) fn predicates_of(tcx: TyCtxt<'_>, def_id: DefId) -> ty::GenericPredicates<'_> {
    let mut result = tcx.explicit_predicates_of(def_id);
    debug!("predicates_of: explicit_predicates_of({:?}) = {:?}", def_id, result);

    let inferred_outlives = tcx.inferred_outlives_of(def_id);
    if !inferred_outlives.is_empty() {
        debug!("predicates_of: inferred_outlives_of({:?}) = {:?}", def_id, inferred_outlives,);
        let inferred_outlives_iter =
            inferred_outlives.iter().map(|(clause, span)| ((*clause).upcast(tcx), *span));
        if result.predicates.is_empty() {
            result.predicates = tcx.arena.alloc_from_iter(inferred_outlives_iter);
        } else {
            result.predicates = tcx.arena.alloc_from_iter(
                result.predicates.into_iter().copied().chain(inferred_outlives_iter),
            );
        }
    }

    if tcx.is_trait(def_id) {
        // For traits, add `Self: Trait` predicate. This is
        // not part of the predicates that a user writes, but it
        // is something that one must prove in order to invoke a
        // method or project an associated type.
        //
        // In the chalk setup, this predicate is not part of the
        // "predicates" for a trait item. But it is useful in
        // rustc because if you directly (e.g.) invoke a trait
        // method like `Trait::method(...)`, you must naturally
        // prove that the trait applies to the types that were
        // used, and adding the predicate into this list ensures
        // that this is done.
        //
        // We use a DUMMY_SP here as a way to signal trait bounds that come
        // from the trait itself that *shouldn't* be shown as the source of
        // an obligation and instead be skipped. Otherwise we'd use
        // `tcx.def_span(def_id);`
        let span = DUMMY_SP;

        result.predicates = tcx.arena.alloc_from_iter(
            result
                .predicates
                .iter()
                .copied()
                .chain(std::iter::once((ty::TraitRef::identity(tcx, def_id).upcast(tcx), span))),
        );
    }

    debug!("predicates_of({:?}) = {:?}", def_id, result);
    result
}

/// Returns a list of user-specified type predicates for the definition with ID `def_id`.
/// N.B., this does not include any implied/inferred constraints.
#[instrument(level = "trace", skip(tcx), ret)]
fn gather_explicit_predicates_of(tcx: TyCtxt<'_>, def_id: LocalDefId) -> ty::GenericPredicates<'_> {
    use rustc_hir::*;
    use rustc_middle::ty::Ty;

    match tcx.opt_rpitit_info(def_id.to_def_id()) {
        Some(ImplTraitInTraitData::Trait { fn_def_id, .. }) => {
            let mut predicates = Vec::new();

            // RPITITs should inherit the predicates of their parent. This is
            // both to ensure that the RPITITs are only instantiated when the
            // parent predicates would hold, and also so that the param-env
            // inherits these predicates as assumptions.
            let identity_args = ty::GenericArgs::identity_for_item(tcx, def_id);
            predicates
                .extend(tcx.explicit_predicates_of(fn_def_id).instantiate_own(tcx, identity_args));

            // We also install bidirectional outlives predicates for the RPITIT
            // to keep the duplicates lifetimes from opaque lowering in sync.
            // We only need to compute bidirectional outlives for the duplicated
            // opaque lifetimes, which explains the slicing below.
            compute_bidirectional_outlives_predicates(
                tcx,
                &tcx.generics_of(def_id.to_def_id()).own_params
                    [tcx.generics_of(fn_def_id).own_params.len()..],
                &mut predicates,
            );

            return ty::GenericPredicates {
                parent: Some(tcx.parent(def_id.to_def_id())),
                predicates: tcx.arena.alloc_from_iter(predicates),
                effects_min_tys: ty::List::empty(),
            };
        }

        Some(ImplTraitInTraitData::Impl { fn_def_id }) => {
            let assoc_item = tcx.associated_item(def_id);
            let trait_assoc_predicates =
                tcx.explicit_predicates_of(assoc_item.trait_item_def_id.unwrap());

            let impl_assoc_identity_args = ty::GenericArgs::identity_for_item(tcx, def_id);
            let impl_def_id = tcx.parent(fn_def_id);
            let impl_trait_ref_args =
                tcx.impl_trait_ref(impl_def_id).unwrap().instantiate_identity().args;

            let impl_assoc_args =
                impl_assoc_identity_args.rebase_onto(tcx, impl_def_id, impl_trait_ref_args);

            let impl_predicates = trait_assoc_predicates.instantiate_own(tcx, impl_assoc_args);

            return ty::GenericPredicates {
                parent: Some(impl_def_id),
                predicates: tcx.arena.alloc_from_iter(impl_predicates),
                effects_min_tys: ty::List::empty(),
            };
        }

        None => {}
    }

    let hir_id = tcx.local_def_id_to_hir_id(def_id);
    let node = tcx.hir_node(hir_id);

    if let Some(sig) = node.fn_sig()
        && let Some(sig_id) = sig.decl.opt_delegation_sig_id()
    {
        return inherit_predicates_for_delegation_item(tcx, def_id, sig_id);
    }

    let mut is_trait = None;
    let mut is_default_impl_trait = None;

    let icx = ItemCtxt::new(tcx, def_id);

    const NO_GENERICS: &hir::Generics<'_> = hir::Generics::empty();

    // We use an `IndexSet` to preserve order of insertion.
    // Preserving the order of insertion is important here so as not to break UI tests.
    let mut predicates: FxIndexSet<(ty::Clause<'_>, Span)> = FxIndexSet::default();
    let mut effects_min_tys = Vec::new();

    let hir_generics = node.generics().unwrap_or(NO_GENERICS);
    if let Node::Item(item) = node {
        match item.kind {
            ItemKind::Impl(impl_) => {
                if impl_.defaultness.is_default() {
                    is_default_impl_trait = tcx
                        .impl_trait_ref(def_id)
                        .map(|t| ty::Binder::dummy(t.instantiate_identity()));
                }
            }

            ItemKind::Trait(_, _, _, self_bounds, ..) | ItemKind::TraitAlias(_, self_bounds) => {
                is_trait = Some(self_bounds);
            }
            _ => {}
        }
    };

    let generics = tcx.generics_of(def_id);

    // Below we'll consider the bounds on the type parameters (including `Self`)
    // and the explicit where-clauses, but to get the full set of predicates
    // on a trait we must also consider the bounds that follow the trait's name,
    // like `trait Foo: A + B + C`.
    if let Some(self_bounds) = is_trait {
        let mut bounds = Bounds::default();
        icx.lowerer().lower_bounds(
            tcx.types.self_param,
            self_bounds,
            &mut bounds,
            ty::List::empty(),
            PredicateFilter::All,
        );
        predicates.extend(bounds.clauses(tcx));
        effects_min_tys.extend(bounds.effects_min_tys());
    }

    // In default impls, we can assume that the self type implements
    // the trait. So in:
    //
    //     default impl Foo for Bar { .. }
    //
    // we add a default where clause `Bar: Foo`. We do a similar thing for traits
    // (see below). Recall that a default impl is not itself an impl, but rather a
    // set of defaults that can be incorporated into another impl.
    if let Some(trait_ref) = is_default_impl_trait {
        predicates.insert((trait_ref.upcast(tcx), tcx.def_span(def_id)));
    }

    // Add implicit predicates that should be treated as if the user has written them,
    // including the implicit `T: Sized` for all generic parameters, and `ConstArgHasType`
    // for const params.
    for param in hir_generics.params {
        match param.kind {
            GenericParamKind::Lifetime { .. } => (),
            GenericParamKind::Type { .. } => {
                let param_ty = icx.lowerer().lower_ty_param(param.hir_id);
                let mut bounds = Bounds::default();
                // Params are implicitly sized unless a `?Sized` bound is found
                icx.lowerer().add_sized_bound(
                    &mut bounds,
                    param_ty,
                    &[],
                    Some((param.def_id, hir_generics.predicates)),
                    param.span,
                );
                trace!(?bounds);
                predicates.extend(bounds.clauses(tcx));
                trace!(?predicates);
            }
            hir::GenericParamKind::Const { .. } => {
                let ct_ty = tcx
                    .type_of(param.def_id.to_def_id())
                    .no_bound_vars()
                    .expect("const parameters cannot be generic");
                let ct = icx.lowerer().lower_const_param(param.hir_id);
                predicates
                    .insert((ty::ClauseKind::ConstArgHasType(ct, ct_ty).upcast(tcx), param.span));
            }
        }
    }

    trace!(?predicates);
    // Add inline `<T: Foo>` bounds and bounds in the where clause.
    for predicate in hir_generics.predicates {
        match predicate {
            hir::WherePredicate::BoundPredicate(bound_pred) => {
                let ty = icx.lowerer().lower_ty_maybe_return_type_notation(bound_pred.bounded_ty);

                let bound_vars = tcx.late_bound_vars(bound_pred.hir_id);
                // Keep the type around in a dummy predicate, in case of no bounds.
                // That way, `where Ty:` is not a complete noop (see #53696) and `Ty`
                // is still checked for WF.
                if bound_pred.bounds.is_empty() {
                    if let ty::Param(_) = ty.kind() {
                        // This is a `where T:`, which can be in the HIR from the
                        // transformation that moves `?Sized` to `T`'s declaration.
                        // We can skip the predicate because type parameters are
                        // trivially WF, but also we *should*, to avoid exposing
                        // users who never wrote `where Type:,` themselves, to
                        // compiler/tooling bugs from not handling WF predicates.
                    } else {
                        let span = bound_pred.bounded_ty.span;
                        let predicate = ty::Binder::bind_with_vars(
                            ty::ClauseKind::WellFormed(ty.into()),
                            bound_vars,
                        );
                        predicates.insert((predicate.upcast(tcx), span));
                    }
                }

                let mut bounds = Bounds::default();
                icx.lowerer().lower_bounds(
                    ty,
                    bound_pred.bounds,
                    &mut bounds,
                    bound_vars,
                    PredicateFilter::All,
                );
                predicates.extend(bounds.clauses(tcx));
                effects_min_tys.extend(bounds.effects_min_tys());
            }

            hir::WherePredicate::RegionPredicate(region_pred) => {
                let r1 = icx
                    .lowerer()
                    .lower_lifetime(region_pred.lifetime, RegionInferReason::RegionPredicate);
                predicates.extend(region_pred.bounds.iter().map(|bound| {
                    let (r2, span) = match bound {
                        hir::GenericBound::Outlives(lt) => (
                            icx.lowerer().lower_lifetime(lt, RegionInferReason::RegionPredicate),
                            lt.ident.span,
                        ),
                        bound => {
                            span_bug!(
                                bound.span(),
                                "lifetime param bounds must be outlives, but found {bound:?}"
                            )
                        }
                    };
                    let pred =
                        ty::ClauseKind::RegionOutlives(ty::OutlivesPredicate(r1, r2)).upcast(tcx);
                    (pred, span)
                }))
            }

            hir::WherePredicate::EqPredicate(..) => {
                // FIXME(#20041)
            }
        }
    }

    if tcx.features().generic_const_exprs {
        predicates.extend(const_evaluatable_predicates_of(tcx, def_id));
    }

    let mut predicates: Vec<_> = predicates.into_iter().collect();

    // Subtle: before we store the predicates into the tcx, we
    // sort them so that predicates like `T: Foo<Item=U>` come
    // before uses of `U`. This avoids false ambiguity errors
    // in trait checking. See `setup_constraining_predicates`
    // for details.
    if let Node::Item(&Item { kind: ItemKind::Impl { .. }, .. }) = node {
        let self_ty = tcx.type_of(def_id).instantiate_identity();
        let trait_ref = tcx.impl_trait_ref(def_id).map(ty::EarlyBinder::instantiate_identity);
        cgp::setup_constraining_predicates(
            tcx,
            &mut predicates,
            trait_ref,
            &mut cgp::parameters_for_impl(tcx, self_ty, trait_ref),
        );
    }

    // Opaque types duplicate some of their generic parameters.
    // We create bi-directional Outlives predicates between the original
    // and the duplicated parameter, to ensure that they do not get out of sync.
    if let Node::OpaqueTy(..) = node {
        let opaque_ty_node = tcx.parent_hir_node(hir_id);
        let Node::Ty(&hir::Ty { kind: TyKind::OpaqueDef(_, lifetimes), .. }) = opaque_ty_node
        else {
            bug!("unexpected {opaque_ty_node:?}")
        };
        debug!(?lifetimes);

        compute_bidirectional_outlives_predicates(tcx, &generics.own_params, &mut predicates);
        debug!(?predicates);
    }

    // add `Self::Effects: Compat<HOST>` to ensure non-const impls don't get called
    // in const contexts.
    if let Node::TraitItem(&TraitItem { kind: TraitItemKind::Fn(..), .. }) = node
        && let Some(host_effect_index) = generics.host_effect_index
    {
        let parent = generics.parent.unwrap();
        let Some(assoc_def_id) = tcx.associated_type_for_effects(parent) else {
            bug!("associated_type_for_effects returned None when there is host effect in generics");
        };
        let effects =
            Ty::new_projection(tcx, assoc_def_id, ty::GenericArgs::identity_for_item(tcx, parent));
        let param = generics.param_at(host_effect_index, tcx);
        let span = tcx.def_span(param.def_id);
        let host = ty::Const::new_param(tcx, ty::ParamConst::for_def(param));
        let compat = tcx.require_lang_item(LangItem::EffectsCompat, Some(span));
        let trait_ref =
            ty::TraitRef::new(tcx, compat, [ty::GenericArg::from(effects), host.into()]);
        predicates.push((ty::Binder::dummy(trait_ref).upcast(tcx), span));
    }

    ty::GenericPredicates {
        parent: generics.parent,
        predicates: tcx.arena.alloc_from_iter(predicates),
        effects_min_tys: tcx.mk_type_list(&effects_min_tys),
    }
}

/// Opaques have duplicated lifetimes and we need to compute bidirectional outlives predicates to
/// enforce that these lifetimes stay in sync.
fn compute_bidirectional_outlives_predicates<'tcx>(
    tcx: TyCtxt<'tcx>,
    opaque_own_params: &[ty::GenericParamDef],
    predicates: &mut Vec<(ty::Clause<'tcx>, Span)>,
) {
    for param in opaque_own_params {
        let orig_lifetime = tcx.map_opaque_lifetime_to_parent_lifetime(param.def_id.expect_local());
        if let ty::ReEarlyParam(..) = *orig_lifetime {
            let dup_lifetime = ty::Region::new_early_param(tcx, ty::EarlyParamRegion {
                index: param.index,
                name: param.name,
            });
            let span = tcx.def_span(param.def_id);
            predicates.push((
                ty::ClauseKind::RegionOutlives(ty::OutlivesPredicate(orig_lifetime, dup_lifetime))
                    .upcast(tcx),
                span,
            ));
            predicates.push((
                ty::ClauseKind::RegionOutlives(ty::OutlivesPredicate(dup_lifetime, orig_lifetime))
                    .upcast(tcx),
                span,
            ));
        }
    }
}

fn const_evaluatable_predicates_of(
    tcx: TyCtxt<'_>,
    def_id: LocalDefId,
) -> FxIndexSet<(ty::Clause<'_>, Span)> {
    struct ConstCollector<'tcx> {
        tcx: TyCtxt<'tcx>,
        preds: FxIndexSet<(ty::Clause<'tcx>, Span)>,
    }

    impl<'tcx> intravisit::Visitor<'tcx> for ConstCollector<'tcx> {
        fn visit_anon_const(&mut self, c: &'tcx hir::AnonConst) {
            let ct = ty::Const::from_anon_const(self.tcx, c.def_id);
            if let ty::ConstKind::Unevaluated(_) = ct.kind() {
                let span = self.tcx.def_span(c.def_id);
                self.preds.insert((ty::ClauseKind::ConstEvaluatable(ct).upcast(self.tcx), span));
            }
        }

        fn visit_const_param_default(&mut self, _param: HirId, _ct: &'tcx hir::ConstArg<'tcx>) {
            // Do not look into const param defaults,
            // these get checked when they are actually instantiated.
            //
            // We do not want the following to error:
            //
            //     struct Foo<const N: usize, const M: usize = { N + 1 }>;
            //     struct Bar<const N: usize>(Foo<N, 3>);
        }
    }

    let hir_id = tcx.local_def_id_to_hir_id(def_id);
    let node = tcx.hir_node(hir_id);

    let mut collector = ConstCollector { tcx, preds: FxIndexSet::default() };
    if let hir::Node::Item(item) = node
        && let hir::ItemKind::Impl(impl_) = item.kind
    {
        if let Some(of_trait) = &impl_.of_trait {
            debug!("const_evaluatable_predicates_of({:?}): visit impl trait_ref", def_id);
            collector.visit_trait_ref(of_trait);
        }

        debug!("const_evaluatable_predicates_of({:?}): visit_self_ty", def_id);
        collector.visit_ty(impl_.self_ty);
    }

    if let Some(generics) = node.generics() {
        debug!("const_evaluatable_predicates_of({:?}): visit_generics", def_id);
        collector.visit_generics(generics);
    }

    if let Some(fn_sig) = tcx.hir().fn_sig_by_hir_id(hir_id) {
        debug!("const_evaluatable_predicates_of({:?}): visit_fn_decl", def_id);
        collector.visit_fn_decl(fn_sig.decl);
    }
    debug!("const_evaluatable_predicates_of({:?}) = {:?}", def_id, collector.preds);

    collector.preds
}

pub(super) fn explicit_predicates_of<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: LocalDefId,
) -> ty::GenericPredicates<'tcx> {
    let def_kind = tcx.def_kind(def_id);
    if matches!(def_kind, DefKind::AnonConst)
        && tcx.features().generic_const_exprs
        && let Some(defaulted_param_def_id) =
            tcx.hir().opt_const_param_default_param_def_id(tcx.local_def_id_to_hir_id(def_id))
    {
        // In `generics_of` we set the generics' parent to be our parent's parent which means that
        // we lose out on the predicates of our actual parent if we dont return those predicates here.
        // (See comment in `generics_of` for more information on why the parent shenanigans is necessary)
        //
        // struct Foo<T, const N: usize = { <T as Trait>::ASSOC }>(T) where T: Trait;
        //        ^^^                     ^^^^^^^^^^^^^^^^^^^^^^^ the def id we are calling
        //        ^^^                                             explicit_predicates_of on
        //        parent item we dont have set as the
        //        parent of generics returned by `generics_of`
        //
        // In the above code we want the anon const to have predicates in its param env for `T: Trait`
        // and we would be calling `explicit_predicates_of(Foo)` here
        let parent_def_id = tcx.local_parent(def_id);
        let parent_preds = tcx.explicit_predicates_of(parent_def_id);

        // If we dont filter out `ConstArgHasType` predicates then every single defaulted const parameter
        // will ICE because of #106994. FIXME(generic_const_exprs): remove this when a more general solution
        // to #106994 is implemented.
        let filtered_predicates = parent_preds
            .predicates
            .into_iter()
            .filter(|(pred, _)| {
                if let ty::ClauseKind::ConstArgHasType(ct, _) = pred.kind().skip_binder() {
                    match ct.kind() {
                        ty::ConstKind::Param(param_const) => {
                            let defaulted_param_idx = tcx
                                .generics_of(parent_def_id)
                                .param_def_id_to_index[&defaulted_param_def_id.to_def_id()];
                            param_const.index < defaulted_param_idx
                        }
                        _ => bug!(
                            "`ConstArgHasType` in `predicates_of`\
                                 that isn't a `Param` const"
                        ),
                    }
                } else {
                    true
                }
            })
            .cloned();
        return GenericPredicates {
            parent: parent_preds.parent,
            predicates: { tcx.arena.alloc_from_iter(filtered_predicates) },
            effects_min_tys: parent_preds.effects_min_tys,
        };
    }
    gather_explicit_predicates_of(tcx, def_id)
}

/// Ensures that the super-predicates of the trait with a `DefId`
/// of `trait_def_id` are lowered and stored. This also ensures that
/// the transitive super-predicates are lowered.
pub(super) fn explicit_super_predicates_of<'tcx>(
    tcx: TyCtxt<'tcx>,
    trait_def_id: LocalDefId,
) -> ty::EarlyBinder<'tcx, &'tcx [(ty::Clause<'tcx>, Span)]> {
    implied_predicates_with_filter(tcx, trait_def_id.to_def_id(), PredicateFilter::SelfOnly)
}

pub(super) fn explicit_supertraits_containing_assoc_item<'tcx>(
    tcx: TyCtxt<'tcx>,
    (trait_def_id, assoc_name): (DefId, Ident),
) -> ty::EarlyBinder<'tcx, &'tcx [(ty::Clause<'tcx>, Span)]> {
    implied_predicates_with_filter(tcx, trait_def_id, PredicateFilter::SelfThatDefines(assoc_name))
}

pub(super) fn explicit_implied_predicates_of<'tcx>(
    tcx: TyCtxt<'tcx>,
    trait_def_id: LocalDefId,
) -> ty::EarlyBinder<'tcx, &'tcx [(ty::Clause<'tcx>, Span)]> {
    implied_predicates_with_filter(
        tcx,
        trait_def_id.to_def_id(),
        if tcx.is_trait_alias(trait_def_id.to_def_id()) {
            PredicateFilter::All
        } else {
            PredicateFilter::SelfAndAssociatedTypeBounds
        },
    )
}

/// Ensures that the super-predicates of the trait with a `DefId`
/// of `trait_def_id` are lowered and stored. This also ensures that
/// the transitive super-predicates are lowered.
pub(super) fn implied_predicates_with_filter<'tcx>(
    tcx: TyCtxt<'tcx>,
    trait_def_id: DefId,
    filter: PredicateFilter,
) -> ty::EarlyBinder<'tcx, &'tcx [(ty::Clause<'tcx>, Span)]> {
    let Some(trait_def_id) = trait_def_id.as_local() else {
        // if `assoc_name` is None, then the query should've been redirected to an
        // external provider
        assert_matches!(filter, PredicateFilter::SelfThatDefines(_));
        return tcx.explicit_super_predicates_of(trait_def_id);
    };

    let Node::Item(item) = tcx.hir_node_by_def_id(trait_def_id) else {
        bug!("trait_def_id {trait_def_id:?} is not an item");
    };

    let (generics, superbounds) = match item.kind {
        hir::ItemKind::Trait(.., generics, supertraits, _) => (generics, supertraits),
        hir::ItemKind::TraitAlias(generics, supertraits) => (generics, supertraits),
        _ => span_bug!(item.span, "super_predicates invoked on non-trait"),
    };

    let icx = ItemCtxt::new(tcx, trait_def_id);

    let self_param_ty = tcx.types.self_param;
    let mut bounds = Bounds::default();
    icx.lowerer().lower_bounds(self_param_ty, superbounds, &mut bounds, ty::List::empty(), filter);

    let where_bounds_that_match = icx.probe_ty_param_bounds_in_generics(
        generics,
        item.owner_id.def_id,
        self_param_ty,
        filter,
    );

    // Combine the two lists to form the complete set of superbounds:
    let implied_bounds =
        &*tcx.arena.alloc_from_iter(bounds.clauses(tcx).chain(where_bounds_that_match));
    debug!(?implied_bounds);

    // Now require that immediate supertraits are lowered, which will, in
    // turn, reach indirect supertraits, so we detect cycles now instead of
    // overflowing during elaboration. Same for implied predicates, which
    // make sure we walk into associated type bounds.
    match filter {
        PredicateFilter::SelfOnly => {
            for &(pred, span) in implied_bounds {
                debug!("superbound: {:?}", pred);
                if let ty::ClauseKind::Trait(bound) = pred.kind().skip_binder()
                    && bound.polarity == ty::PredicatePolarity::Positive
                {
                    tcx.at(span).explicit_super_predicates_of(bound.def_id());
                }
            }
        }
        PredicateFilter::SelfAndAssociatedTypeBounds => {
            for &(pred, span) in implied_bounds {
                debug!("superbound: {:?}", pred);
                if let ty::ClauseKind::Trait(bound) = pred.kind().skip_binder()
                    && bound.polarity == ty::PredicatePolarity::Positive
                {
                    tcx.at(span).explicit_implied_predicates_of(bound.def_id());
                }
            }
        }
        _ => {}
    }

    assert_only_contains_predicates_from(filter, implied_bounds, tcx.types.self_param);

    ty::EarlyBinder::bind(implied_bounds)
}

// Make sure when elaborating supertraits, probing for associated types, etc.,
// we really truly are elaborating clauses that have `Self` as their self type.
// This is very important since downstream code relies on this being correct.
pub(super) fn assert_only_contains_predicates_from<'tcx>(
    filter: PredicateFilter,
    bounds: &'tcx [(ty::Clause<'tcx>, Span)],
    ty: Ty<'tcx>,
) {
    if !cfg!(debug_assertions) {
        return;
    }

    match filter {
        PredicateFilter::SelfOnly | PredicateFilter::SelfThatDefines(_) => {
            for (clause, _) in bounds {
                match clause.kind().skip_binder() {
                    ty::ClauseKind::Trait(trait_predicate) => {
                        assert_eq!(
                            trait_predicate.self_ty(),
                            ty,
                            "expected `Self` predicate when computing `{filter:?}` implied bounds: {clause:?}"
                        );
                    }
                    ty::ClauseKind::Projection(projection_predicate) => {
                        assert_eq!(
                            projection_predicate.self_ty(),
                            ty,
                            "expected `Self` predicate when computing `{filter:?}` implied bounds: {clause:?}"
                        );
                    }
                    ty::ClauseKind::TypeOutlives(outlives_predicate) => {
                        assert_eq!(
                            outlives_predicate.0, ty,
                            "expected `Self` predicate when computing `{filter:?}` implied bounds: {clause:?}"
                        );
                    }

                    ty::ClauseKind::RegionOutlives(_)
                    | ty::ClauseKind::ConstArgHasType(_, _)
                    | ty::ClauseKind::WellFormed(_)
                    | ty::ClauseKind::ConstEvaluatable(_) => {
                        bug!(
                            "unexpected non-`Self` predicate when computing `{filter:?}` implied bounds: {clause:?}"
                        );
                    }
                }
            }
        }
        PredicateFilter::All | PredicateFilter::SelfAndAssociatedTypeBounds => {}
    }
}

/// Returns the predicates defined on `item_def_id` of the form
/// `X: Foo` where `X` is the type parameter `def_id`.
#[instrument(level = "trace", skip(tcx))]
pub(super) fn type_param_predicates<'tcx>(
    tcx: TyCtxt<'tcx>,
    (item_def_id, def_id, assoc_name): (LocalDefId, LocalDefId, Ident),
) -> ty::EarlyBinder<'tcx, &'tcx [(ty::Clause<'tcx>, Span)]> {
    use rustc_hir::*;
    use rustc_middle::ty::Ty;

    // In the HIR, bounds can derive from two places. Either
    // written inline like `<T: Foo>` or in a where-clause like
    // `where T: Foo`.

    let param_id = tcx.local_def_id_to_hir_id(def_id);
    let param_owner = tcx.hir().ty_param_owner(def_id);
    let generics = tcx.generics_of(param_owner);
    let index = generics.param_def_id_to_index[&def_id.to_def_id()];
    let ty = Ty::new_param(tcx, index, tcx.hir().ty_param_name(def_id));

    // Don't look for bounds where the type parameter isn't in scope.
    let parent = if item_def_id == param_owner {
        None
    } else {
        tcx.generics_of(item_def_id).parent.map(|def_id| def_id.expect_local())
    };

    let result = if let Some(parent) = parent {
        let icx = ItemCtxt::new(tcx, parent);
        icx.probe_ty_param_bounds(DUMMY_SP, def_id, assoc_name)
    } else {
        ty::EarlyBinder::bind(&[] as &[_])
    };
    let mut extend = None;

    let item_hir_id = tcx.local_def_id_to_hir_id(item_def_id);

    let hir_node = tcx.hir_node(item_hir_id);
    let Some(hir_generics) = hir_node.generics() else {
        return result;
    };
    if let Node::Item(item) = hir_node
        && let ItemKind::Trait(..) = item.kind
        // Implied `Self: Trait` and supertrait bounds.
        && param_id == item_hir_id
    {
        let identity_trait_ref = ty::TraitRef::identity(tcx, item_def_id.to_def_id());
        extend = Some((identity_trait_ref.upcast(tcx), item.span));
    }

    let icx = ItemCtxt::new(tcx, item_def_id);
    let extra_predicates = extend.into_iter().chain(
        icx.probe_ty_param_bounds_in_generics(
            hir_generics,
            def_id,
            ty,
            PredicateFilter::SelfThatDefines(assoc_name),
        )
        .into_iter()
        .filter(|(predicate, _)| match predicate.kind().skip_binder() {
            ty::ClauseKind::Trait(data) => data.self_ty().is_param(index),
            _ => false,
        }),
    );

    ty::EarlyBinder::bind(
        tcx.arena.alloc_from_iter(result.skip_binder().iter().copied().chain(extra_predicates)),
    )
}

impl<'tcx> ItemCtxt<'tcx> {
    /// Finds bounds from `hir::Generics`.
    ///
    /// This requires scanning through the HIR.
    /// We do this to avoid having to lower *all* the bounds, which would create artificial cycles.
    /// Instead, we can only lower the bounds for a type parameter `X` if `X::Foo` is used.
    #[instrument(level = "trace", skip(self, hir_generics))]
    fn probe_ty_param_bounds_in_generics(
        &self,
        hir_generics: &'tcx hir::Generics<'tcx>,
        param_def_id: LocalDefId,
        ty: Ty<'tcx>,
        filter: PredicateFilter,
    ) -> Vec<(ty::Clause<'tcx>, Span)> {
        let mut bounds = Bounds::default();

        for predicate in hir_generics.predicates {
            let hir::WherePredicate::BoundPredicate(predicate) = predicate else {
                continue;
            };

            let bound_ty = if predicate.is_param_bound(param_def_id.to_def_id()) {
                ty
            } else if matches!(filter, PredicateFilter::All) {
                self.lowerer().lower_ty_maybe_return_type_notation(predicate.bounded_ty)
            } else {
                continue;
            };

            let bound_vars = self.tcx.late_bound_vars(predicate.hir_id);
            self.lowerer().lower_bounds(
                bound_ty,
                predicate.bounds,
                &mut bounds,
                bound_vars,
                filter,
            );
        }

        bounds.clauses(self.tcx).collect()
    }
}
