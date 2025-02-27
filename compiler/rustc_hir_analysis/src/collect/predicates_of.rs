use std::assert_matches::assert_matches;

use hir::Node;
use rustc_data_structures::fx::FxIndexSet;
use rustc_hir as hir;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_middle::ty::{
    self, GenericPredicates, ImplTraitInTraitData, Ty, TyCtxt, TypeVisitable, TypeVisitor, Upcast,
};
use rustc_middle::{bug, span_bug};
use rustc_span::{DUMMY_SP, Ident, Span};
use tracing::{debug, instrument, trace};

use super::item_bounds::explicit_item_bounds_with_filter;
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
            ItemKind::Trait(_, _, _, _, self_bounds, ..)
            | ItemKind::TraitAlias(_, _, self_bounds) => {
                is_trait = Some((self_bounds, item.span));
            }
            _ => {}
        }
    };

    if let Node::TraitItem(item) = node {
        let mut bounds = Vec::new();
        icx.lowerer().add_default_trait_item_bounds(item, &mut bounds);
        predicates.extend(bounds);
    }

    let generics = tcx.generics_of(def_id);

    // Below we'll consider the bounds on the type parameters (including `Self`)
    // and the explicit where-clauses, but to get the full set of predicates
    // on a trait we must also consider the bounds that follow the trait's name,
    // like `trait Foo: A + B + C`.
    if let Some((self_bounds, span)) = is_trait {
        let mut bounds = Vec::new();
        icx.lowerer().lower_bounds(
            tcx.types.self_param,
            self_bounds,
            &mut bounds,
            ty::List::empty(),
            PredicateFilter::All,
        );
        icx.lowerer().add_sizedness_bounds(
            &mut bounds,
            tcx.types.self_param,
            self_bounds,
            None,
            Some(def_id),
            span,
        );
        icx.lowerer().add_default_super_traits(
            def_id,
            &mut bounds,
            self_bounds,
            hir_generics,
            span,
        );
        predicates.extend(bounds);
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
                let mut bounds = Vec::new();
                // Implicit bounds are added to type params unless a `?Trait` bound is found
                icx.lowerer().add_sizedness_bounds(
                    &mut bounds,
                    param_ty,
                    &[],
                    Some((param.def_id, hir_generics.predicates)),
                    None,
                    param.span,
                );
                icx.lowerer().add_default_traits(
                    &mut bounds,
                    param_ty,
                    &[],
                    Some((param.def_id, hir_generics.predicates)),
                    param.span,
                );
                trace!(?bounds);
                predicates.extend(bounds);
                trace!(?predicates);
            }
            hir::GenericParamKind::Const { .. } => {
                let param_def_id = param.def_id.to_def_id();
                let ct_ty = tcx.type_of(param_def_id).instantiate_identity();
                let ct = icx.lowerer().lower_const_param(param_def_id, param.hir_id);
                predicates
                    .insert((ty::ClauseKind::ConstArgHasType(ct, ct_ty).upcast(tcx), param.span));
            }
        }
    }

    trace!(?predicates);
    // Add inline `<T: Foo>` bounds and bounds in the where clause.
    for predicate in hir_generics.predicates {
        match predicate.kind {
            hir::WherePredicateKind::BoundPredicate(bound_pred) => {
                let ty = icx.lowerer().lower_ty_maybe_return_type_notation(bound_pred.bounded_ty);

                let bound_vars = tcx.late_bound_vars(predicate.hir_id);
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

                let mut bounds = Vec::new();
                icx.lowerer().lower_bounds(
                    ty,
                    bound_pred.bounds,
                    &mut bounds,
                    bound_vars,
                    PredicateFilter::All,
                );
                predicates.extend(bounds);
            }

            hir::WherePredicateKind::RegionPredicate(region_pred) => {
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

            hir::WherePredicateKind::EqPredicate(..) => {
                // FIXME(#20041)
            }
        }
    }

    if tcx.features().generic_const_exprs() {
        predicates.extend(const_evaluatable_predicates_of(tcx, def_id, &predicates));
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
        compute_bidirectional_outlives_predicates(tcx, &generics.own_params, &mut predicates);
        debug!(?predicates);
    }

    ty::GenericPredicates {
        parent: generics.parent,
        predicates: tcx.arena.alloc_from_iter(predicates),
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
        if let ty::ReEarlyParam(..) = orig_lifetime.kind() {
            let dup_lifetime = ty::Region::new_early_param(
                tcx,
                ty::EarlyParamRegion { index: param.index, name: param.name },
            );
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

#[instrument(level = "debug", skip(tcx, predicates), ret)]
fn const_evaluatable_predicates_of<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: LocalDefId,
    predicates: &FxIndexSet<(ty::Clause<'tcx>, Span)>,
) -> FxIndexSet<(ty::Clause<'tcx>, Span)> {
    struct ConstCollector<'tcx> {
        tcx: TyCtxt<'tcx>,
        preds: FxIndexSet<(ty::Clause<'tcx>, Span)>,
    }

    fn is_const_param_default(tcx: TyCtxt<'_>, def: LocalDefId) -> bool {
        let hir_id = tcx.local_def_id_to_hir_id(def);
        let (_, parent_node) = tcx
            .hir_parent_iter(hir_id)
            .skip_while(|(_, n)| matches!(n, Node::ConstArg(..)))
            .next()
            .unwrap();
        matches!(
            parent_node,
            Node::GenericParam(hir::GenericParam { kind: hir::GenericParamKind::Const { .. }, .. })
        )
    }

    impl<'tcx> TypeVisitor<TyCtxt<'tcx>> for ConstCollector<'tcx> {
        fn visit_const(&mut self, c: ty::Const<'tcx>) {
            if let ty::ConstKind::Unevaluated(uv) = c.kind() {
                if is_const_param_default(self.tcx, uv.def.expect_local()) {
                    // Do not look into const param defaults,
                    // these get checked when they are actually instantiated.
                    //
                    // We do not want the following to error:
                    //
                    //     struct Foo<const N: usize, const M: usize = { N + 1 }>;
                    //     struct Bar<const N: usize>(Foo<N, 3>);
                    return;
                }

                let span = self.tcx.def_span(uv.def);
                self.preds.insert((ty::ClauseKind::ConstEvaluatable(c).upcast(self.tcx), span));
            }
        }
    }

    let hir_id = tcx.local_def_id_to_hir_id(def_id);
    let node = tcx.hir_node(hir_id);

    let mut collector = ConstCollector { tcx, preds: FxIndexSet::default() };

    for (clause, _sp) in predicates {
        clause.visit_with(&mut collector);
    }

    if let hir::Node::Item(item) = node
        && let hir::ItemKind::Impl(_) = item.kind
    {
        if let Some(of_trait) = tcx.impl_trait_ref(def_id) {
            debug!("visit impl trait_ref");
            of_trait.instantiate_identity().visit_with(&mut collector);
        }

        debug!("visit self_ty");
        let self_ty = tcx.type_of(def_id);
        self_ty.instantiate_identity().visit_with(&mut collector);
    }

    if let Some(_) = tcx.hir_fn_sig_by_hir_id(hir_id) {
        debug!("visit fn sig");
        let fn_sig = tcx.fn_sig(def_id);
        let fn_sig = fn_sig.instantiate_identity();
        debug!(?fn_sig);
        fn_sig.visit_with(&mut collector);
    }

    collector.preds
}

pub(super) fn trait_explicit_predicates_and_bounds(
    tcx: TyCtxt<'_>,
    def_id: LocalDefId,
) -> ty::GenericPredicates<'_> {
    assert_eq!(tcx.def_kind(def_id), DefKind::Trait);
    gather_explicit_predicates_of(tcx, def_id)
}

pub(super) fn explicit_predicates_of<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: LocalDefId,
) -> ty::GenericPredicates<'tcx> {
    let def_kind = tcx.def_kind(def_id);
    if let DefKind::Trait = def_kind {
        // Remove bounds on associated types from the predicates, they will be
        // returned by `explicit_item_bounds`.
        let predicates_and_bounds = tcx.trait_explicit_predicates_and_bounds(def_id);
        let trait_identity_args = ty::GenericArgs::identity_for_item(tcx, def_id);

        let is_assoc_item_ty = |ty: Ty<'tcx>| {
            // For a predicate from a where clause to become a bound on an
            // associated type:
            // * It must use the identity args of the item.
            //   * We're in the scope of the trait, so we can't name any
            //     parameters of the GAT. That means that all we need to
            //     check are that the args of the projection are the
            //     identity args of the trait.
            // * It must be an associated type for this trait (*not* a
            //   supertrait).
            if let ty::Alias(ty::Projection, projection) = ty.kind() {
                projection.args == trait_identity_args
                    // FIXME(return_type_notation): This check should be more robust
                    && !tcx.is_impl_trait_in_trait(projection.def_id)
                    && tcx.associated_item(projection.def_id).container_id(tcx)
                        == def_id.to_def_id()
            } else {
                false
            }
        };

        let predicates: Vec<_> = predicates_and_bounds
            .predicates
            .iter()
            .copied()
            .filter(|(pred, _)| match pred.kind().skip_binder() {
                ty::ClauseKind::Trait(tr) => !is_assoc_item_ty(tr.self_ty()),
                ty::ClauseKind::Projection(proj) => {
                    !is_assoc_item_ty(proj.projection_term.self_ty())
                }
                ty::ClauseKind::TypeOutlives(outlives) => !is_assoc_item_ty(outlives.0),
                _ => true,
            })
            .collect();
        if predicates.len() == predicates_and_bounds.predicates.len() {
            predicates_and_bounds
        } else {
            ty::GenericPredicates {
                parent: predicates_and_bounds.parent,
                predicates: tcx.arena.alloc_slice(&predicates),
            }
        }
    } else {
        if matches!(def_kind, DefKind::AnonConst)
            && tcx.features().generic_const_exprs()
            && let Some(defaulted_param_def_id) =
                tcx.hir_opt_const_param_default_param_def_id(tcx.local_def_id_to_hir_id(def_id))
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
            };
        }
        gather_explicit_predicates_of(tcx, def_id)
    }
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
    (trait_def_id, assoc_ident): (DefId, Ident),
) -> ty::EarlyBinder<'tcx, &'tcx [(ty::Clause<'tcx>, Span)]> {
    implied_predicates_with_filter(
        tcx,
        trait_def_id,
        PredicateFilter::SelfTraitThatDefines(assoc_ident),
    )
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
        // if `assoc_ident` is None, then the query should've been redirected to an
        // external provider
        assert_matches!(filter, PredicateFilter::SelfTraitThatDefines(_));
        return tcx.explicit_super_predicates_of(trait_def_id);
    };

    let Node::Item(item) = tcx.hir_node_by_def_id(trait_def_id) else {
        bug!("trait_def_id {trait_def_id:?} is not an item");
    };

    let (generics, superbounds) = match item.kind {
        hir::ItemKind::Trait(.., generics, supertraits, _) => (generics, supertraits),
        hir::ItemKind::TraitAlias(_, generics, supertraits) => (generics, supertraits),
        _ => span_bug!(item.span, "super_predicates invoked on non-trait"),
    };

    let icx = ItemCtxt::new(tcx, trait_def_id);

    let self_param_ty = tcx.types.self_param;
    let mut bounds = Vec::new();
    icx.lowerer().lower_bounds(self_param_ty, superbounds, &mut bounds, ty::List::empty(), filter);
    match filter {
        PredicateFilter::All
        | PredicateFilter::SelfOnly
        | PredicateFilter::SelfTraitThatDefines(_)
        | PredicateFilter::SelfAndAssociatedTypeBounds => {
            icx.lowerer().add_default_super_traits(
                trait_def_id,
                &mut bounds,
                superbounds,
                generics,
                item.span,
            );
        }
        //`ConstIfConst` is only interested in `~const` bounds.
        PredicateFilter::ConstIfConst | PredicateFilter::SelfConstIfConst => {}
    }

    let where_bounds_that_match =
        icx.probe_ty_param_bounds_in_generics(generics, item.owner_id.def_id, filter);

    // Combine the two lists to form the complete set of superbounds:
    let implied_bounds =
        &*tcx.arena.alloc_from_iter(bounds.into_iter().chain(where_bounds_that_match));
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
        PredicateFilter::All | PredicateFilter::SelfAndAssociatedTypeBounds => {
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
// we really truly are elaborating clauses that have `ty` as their self type.
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
        PredicateFilter::SelfOnly => {
            for (clause, _) in bounds {
                match clause.kind().skip_binder() {
                    ty::ClauseKind::Trait(trait_predicate) => {
                        assert_eq!(
                            trait_predicate.self_ty(),
                            ty,
                            "expected `Self` predicate when computing \
                            `{filter:?}` implied bounds: {clause:?}"
                        );
                    }
                    ty::ClauseKind::Projection(projection_predicate) => {
                        assert_eq!(
                            projection_predicate.self_ty(),
                            ty,
                            "expected `Self` predicate when computing \
                            `{filter:?}` implied bounds: {clause:?}"
                        );
                    }
                    ty::ClauseKind::TypeOutlives(outlives_predicate) => {
                        assert_eq!(
                            outlives_predicate.0, ty,
                            "expected `Self` predicate when computing \
                            `{filter:?}` implied bounds: {clause:?}"
                        );
                    }
                    ty::ClauseKind::HostEffect(host_effect_predicate) => {
                        assert_eq!(
                            host_effect_predicate.self_ty(),
                            ty,
                            "expected `Self` predicate when computing \
                            `{filter:?}` implied bounds: {clause:?}"
                        );
                    }

                    ty::ClauseKind::RegionOutlives(_)
                    | ty::ClauseKind::ConstArgHasType(_, _)
                    | ty::ClauseKind::WellFormed(_)
                    | ty::ClauseKind::ConstEvaluatable(_) => {
                        bug!(
                            "unexpected non-`Self` predicate when computing \
                            `{filter:?}` implied bounds: {clause:?}"
                        );
                    }
                }
            }
        }
        PredicateFilter::SelfTraitThatDefines(_) => {
            for (clause, _) in bounds {
                match clause.kind().skip_binder() {
                    ty::ClauseKind::Trait(trait_predicate) => {
                        assert_eq!(
                            trait_predicate.self_ty(),
                            ty,
                            "expected `Self` predicate when computing \
                            `{filter:?}` implied bounds: {clause:?}"
                        );
                    }

                    ty::ClauseKind::Projection(_)
                    | ty::ClauseKind::TypeOutlives(_)
                    | ty::ClauseKind::RegionOutlives(_)
                    | ty::ClauseKind::ConstArgHasType(_, _)
                    | ty::ClauseKind::WellFormed(_)
                    | ty::ClauseKind::ConstEvaluatable(_)
                    | ty::ClauseKind::HostEffect(..) => {
                        bug!(
                            "unexpected non-`Self` predicate when computing \
                            `{filter:?}` implied bounds: {clause:?}"
                        );
                    }
                }
            }
        }
        PredicateFilter::ConstIfConst => {
            for (clause, _) in bounds {
                match clause.kind().skip_binder() {
                    ty::ClauseKind::HostEffect(ty::HostEffectPredicate {
                        trait_ref: _,
                        constness: ty::BoundConstness::Maybe,
                    }) => {}
                    _ => {
                        bug!(
                            "unexpected non-`HostEffect` predicate when computing \
                            `{filter:?}` implied bounds: {clause:?}"
                        );
                    }
                }
            }
        }
        PredicateFilter::SelfConstIfConst => {
            for (clause, _) in bounds {
                match clause.kind().skip_binder() {
                    ty::ClauseKind::HostEffect(pred) => {
                        assert_eq!(
                            pred.constness,
                            ty::BoundConstness::Maybe,
                            "expected `~const` predicate when computing `{filter:?}` \
                            implied bounds: {clause:?}",
                        );
                        assert_eq!(
                            pred.trait_ref.self_ty(),
                            ty,
                            "expected `Self` predicate when computing `{filter:?}` \
                            implied bounds: {clause:?}"
                        );
                    }
                    _ => {
                        bug!(
                            "unexpected non-`HostEffect` predicate when computing \
                            `{filter:?}` implied bounds: {clause:?}"
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
    (item_def_id, def_id, assoc_ident): (LocalDefId, LocalDefId, Ident),
) -> ty::EarlyBinder<'tcx, &'tcx [(ty::Clause<'tcx>, Span)]> {
    match tcx.opt_rpitit_info(item_def_id.to_def_id()) {
        Some(ty::ImplTraitInTraitData::Trait { opaque_def_id, .. }) => {
            return tcx.type_param_predicates((opaque_def_id.expect_local(), def_id, assoc_ident));
        }
        Some(ty::ImplTraitInTraitData::Impl { .. }) => {
            unreachable!("should not be lowering bounds on RPITIT in impl")
        }
        None => {}
    }

    // In the HIR, bounds can derive from two places. Either
    // written inline like `<T: Foo>` or in a where-clause like
    // `where T: Foo`.

    let param_id = tcx.local_def_id_to_hir_id(def_id);
    let param_owner = tcx.hir_ty_param_owner(def_id);

    // Don't look for bounds where the type parameter isn't in scope.
    let parent = if item_def_id == param_owner {
        // FIXME: Shouldn't this be unreachable?
        None
    } else {
        tcx.generics_of(item_def_id).parent.map(|def_id| def_id.expect_local())
    };

    let result = if let Some(parent) = parent {
        let icx = ItemCtxt::new(tcx, parent);
        icx.probe_ty_param_bounds(DUMMY_SP, def_id, assoc_ident)
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
        && let hir::ItemKind::Trait(..) = item.kind
        // Implied `Self: Trait` and supertrait bounds.
        && param_id == item_hir_id
    {
        let identity_trait_ref = ty::TraitRef::identity(tcx, item_def_id.to_def_id());
        extend = Some((identity_trait_ref.upcast(tcx), item.span));
    }

    let icx = ItemCtxt::new(tcx, item_def_id);
    let extra_predicates = extend.into_iter().chain(icx.probe_ty_param_bounds_in_generics(
        hir_generics,
        def_id,
        PredicateFilter::SelfTraitThatDefines(assoc_ident),
    ));

    let bounds =
        &*tcx.arena.alloc_from_iter(result.skip_binder().iter().copied().chain(extra_predicates));

    // Double check that the bounds *only* contain `SelfTy: Trait` preds.
    let self_ty = match tcx.def_kind(def_id) {
        DefKind::TyParam => Ty::new_param(
            tcx,
            tcx.generics_of(item_def_id)
                .param_def_id_to_index(tcx, def_id.to_def_id())
                .expect("expected generic param to be owned by item"),
            tcx.item_name(def_id.to_def_id()),
        ),
        DefKind::Trait | DefKind::TraitAlias => tcx.types.self_param,
        _ => unreachable!(),
    };
    assert_only_contains_predicates_from(
        PredicateFilter::SelfTraitThatDefines(assoc_ident),
        bounds,
        self_ty,
    );

    ty::EarlyBinder::bind(bounds)
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
        filter: PredicateFilter,
    ) -> Vec<(ty::Clause<'tcx>, Span)> {
        let mut bounds = Vec::new();

        for predicate in hir_generics.predicates {
            let hir_id = predicate.hir_id;
            let hir::WherePredicateKind::BoundPredicate(predicate) = predicate.kind else {
                continue;
            };

            match filter {
                _ if predicate.is_param_bound(param_def_id.to_def_id()) => {
                    // Ok
                }
                PredicateFilter::All => {
                    // Ok
                }
                PredicateFilter::SelfOnly
                | PredicateFilter::SelfTraitThatDefines(_)
                | PredicateFilter::SelfConstIfConst
                | PredicateFilter::SelfAndAssociatedTypeBounds => continue,
                PredicateFilter::ConstIfConst => unreachable!(),
            }

            let bound_ty = self.lowerer().lower_ty_maybe_return_type_notation(predicate.bounded_ty);

            let bound_vars = self.tcx.late_bound_vars(hir_id);
            self.lowerer().lower_bounds(
                bound_ty,
                predicate.bounds,
                &mut bounds,
                bound_vars,
                filter,
            );
        }

        bounds
    }
}

pub(super) fn const_conditions<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: LocalDefId,
) -> ty::ConstConditions<'tcx> {
    if !tcx.is_conditionally_const(def_id) {
        bug!("const_conditions invoked for item that is not conditionally const: {def_id:?}");
    }

    match tcx.opt_rpitit_info(def_id.to_def_id()) {
        // RPITITs inherit const conditions of their parent fn
        Some(
            ty::ImplTraitInTraitData::Impl { fn_def_id }
            | ty::ImplTraitInTraitData::Trait { fn_def_id, .. },
        ) => return tcx.const_conditions(fn_def_id),
        None => {}
    }

    let (generics, trait_def_id_and_supertraits, has_parent) = match tcx.hir_node_by_def_id(def_id)
    {
        Node::Item(item) => match item.kind {
            hir::ItemKind::Impl(impl_) => (impl_.generics, None, false),
            hir::ItemKind::Fn { generics, .. } => (generics, None, false),
            hir::ItemKind::Trait(_, _, _, generics, supertraits, _) => {
                (generics, Some((item.owner_id.def_id, supertraits)), false)
            }
            _ => bug!("const_conditions called on wrong item: {def_id:?}"),
        },
        // While associated types are not really const, we do allow them to have `~const`
        // bounds and where clauses. `const_conditions` is responsible for gathering
        // these up so we can check them in `compare_type_predicate_entailment`, and
        // in `HostEffect` goal computation.
        Node::TraitItem(item) => match item.kind {
            hir::TraitItemKind::Fn(_, _) | hir::TraitItemKind::Type(_, _) => {
                (item.generics, None, true)
            }
            _ => bug!("const_conditions called on wrong item: {def_id:?}"),
        },
        Node::ImplItem(item) => match item.kind {
            hir::ImplItemKind::Fn(_, _) | hir::ImplItemKind::Type(_) => {
                (item.generics, None, tcx.is_conditionally_const(tcx.local_parent(def_id)))
            }
            _ => bug!("const_conditions called on wrong item: {def_id:?}"),
        },
        Node::ForeignItem(item) => match item.kind {
            hir::ForeignItemKind::Fn(_, _, generics) => (generics, None, false),
            _ => bug!("const_conditions called on wrong item: {def_id:?}"),
        },
        Node::OpaqueTy(opaque) => match opaque.origin {
            hir::OpaqueTyOrigin::FnReturn { parent, .. } => return tcx.const_conditions(parent),
            hir::OpaqueTyOrigin::AsyncFn { .. } | hir::OpaqueTyOrigin::TyAlias { .. } => {
                unreachable!()
            }
        },
        // N.B. Tuple ctors are unconditionally constant.
        Node::Ctor(hir::VariantData::Tuple { .. }) => return Default::default(),
        _ => bug!("const_conditions called on wrong item: {def_id:?}"),
    };

    let icx = ItemCtxt::new(tcx, def_id);
    let mut bounds = Vec::new();

    for pred in generics.predicates {
        match pred.kind {
            hir::WherePredicateKind::BoundPredicate(bound_pred) => {
                let ty = icx.lowerer().lower_ty_maybe_return_type_notation(bound_pred.bounded_ty);
                let bound_vars = tcx.late_bound_vars(pred.hir_id);
                icx.lowerer().lower_bounds(
                    ty,
                    bound_pred.bounds.iter(),
                    &mut bounds,
                    bound_vars,
                    PredicateFilter::ConstIfConst,
                );
            }
            _ => {}
        }
    }

    if let Some((def_id, supertraits)) = trait_def_id_and_supertraits {
        // We've checked above that the trait is conditionally const.
        bounds.push((
            ty::Binder::dummy(ty::TraitRef::identity(tcx, def_id.to_def_id()))
                .to_host_effect_clause(tcx, ty::BoundConstness::Maybe),
            DUMMY_SP,
        ));

        icx.lowerer().lower_bounds(
            tcx.types.self_param,
            supertraits,
            &mut bounds,
            ty::List::empty(),
            PredicateFilter::ConstIfConst,
        );
    }

    ty::ConstConditions {
        parent: has_parent.then(|| tcx.local_parent(def_id).to_def_id()),
        predicates: tcx.arena.alloc_from_iter(bounds.into_iter().map(|(clause, span)| {
            (
                clause.kind().map_bound(|clause| match clause {
                    ty::ClauseKind::HostEffect(ty::HostEffectPredicate {
                        trait_ref,
                        constness: ty::BoundConstness::Maybe,
                    }) => trait_ref,
                    _ => bug!("converted {clause:?}"),
                }),
                span,
            )
        })),
    }
}

pub(super) fn explicit_implied_const_bounds<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: LocalDefId,
) -> ty::EarlyBinder<'tcx, &'tcx [(ty::PolyTraitRef<'tcx>, Span)]> {
    if !tcx.is_conditionally_const(def_id) {
        bug!(
            "explicit_implied_const_bounds invoked for item that is not conditionally const: {def_id:?}"
        );
    }

    let bounds = match tcx.opt_rpitit_info(def_id.to_def_id()) {
        // RPITIT's bounds are the same as opaque type bounds, but with
        // a projection self type.
        Some(ty::ImplTraitInTraitData::Trait { .. }) => {
            explicit_item_bounds_with_filter(tcx, def_id, PredicateFilter::ConstIfConst)
        }
        Some(ty::ImplTraitInTraitData::Impl { .. }) => {
            span_bug!(tcx.def_span(def_id), "RPITIT in impl should not have item bounds")
        }
        None => match tcx.hir_node_by_def_id(def_id) {
            Node::Item(hir::Item { kind: hir::ItemKind::Trait(..), .. }) => {
                implied_predicates_with_filter(
                    tcx,
                    def_id.to_def_id(),
                    PredicateFilter::SelfConstIfConst,
                )
            }
            Node::TraitItem(hir::TraitItem { kind: hir::TraitItemKind::Type(..), .. })
            | Node::OpaqueTy(_) => {
                explicit_item_bounds_with_filter(tcx, def_id, PredicateFilter::ConstIfConst)
            }
            _ => bug!("explicit_implied_const_bounds called on wrong item: {def_id:?}"),
        },
    };

    bounds.map_bound(|bounds| {
        &*tcx.arena.alloc_from_iter(bounds.iter().copied().map(|(clause, span)| {
            (
                clause.kind().map_bound(|clause| match clause {
                    ty::ClauseKind::HostEffect(ty::HostEffectPredicate {
                        trait_ref,
                        constness: ty::BoundConstness::Maybe,
                    }) => trait_ref,
                    _ => bug!("converted {clause:?}"),
                }),
                span,
            )
        }))
    })
}
