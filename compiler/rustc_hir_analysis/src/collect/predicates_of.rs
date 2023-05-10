use crate::astconv::{AstConv, OnlySelfBounds};
use crate::bounds::Bounds;
use crate::collect::ItemCtxt;
use crate::constrained_generic_params as cgp;
use hir::{HirId, Node};
use rustc_data_structures::fx::FxIndexSet;
use rustc_hir as hir;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::intravisit::{self, Visitor};
use rustc_middle::ty::subst::InternalSubsts;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_middle::ty::{GenericPredicates, ToPredicate};
use rustc_span::symbol::{sym, Ident};
use rustc_span::{Span, DUMMY_SP};

/// Returns a list of all type predicates (explicit and implicit) for the definition with
/// ID `def_id`. This includes all predicates returned by `predicates_defined_on`, plus
/// `Self: Trait` predicates for traits.
pub(super) fn predicates_of(tcx: TyCtxt<'_>, def_id: DefId) -> ty::GenericPredicates<'_> {
    let mut result = tcx.predicates_defined_on(def_id);

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

        let constness = if tcx.has_attr(def_id, sym::const_trait) {
            ty::BoundConstness::ConstIfConst
        } else {
            ty::BoundConstness::NotConst
        };

        let span = rustc_span::DUMMY_SP;
        result.predicates =
            tcx.arena.alloc_from_iter(result.predicates.iter().copied().chain(std::iter::once((
                ty::TraitRef::identity(tcx, def_id).with_constness(constness).to_predicate(tcx),
                span,
            ))));
    }
    debug!("predicates_of(def_id={:?}) = {:?}", def_id, result);
    result
}

/// Returns a list of user-specified type predicates for the definition with ID `def_id`.
/// N.B., this does not include any implied/inferred constraints.
#[instrument(level = "trace", skip(tcx), ret)]
fn gather_explicit_predicates_of(tcx: TyCtxt<'_>, def_id: LocalDefId) -> ty::GenericPredicates<'_> {
    use rustc_hir::*;

    let hir_id = tcx.hir().local_def_id_to_hir_id(def_id);
    let node = tcx.hir().get(hir_id);

    let mut is_trait = None;
    let mut is_default_impl_trait = None;

    // FIXME: Should ItemCtxt take a LocalDefId?
    let icx = ItemCtxt::new(tcx, def_id);

    const NO_GENERICS: &hir::Generics<'_> = hir::Generics::empty();

    // We use an `IndexSet` to preserve order of insertion.
    // Preserving the order of insertion is important here so as not to break UI tests.
    let mut predicates: FxIndexSet<(ty::Predicate<'_>, Span)> = FxIndexSet::default();

    let ast_generics = match node {
        Node::TraitItem(item) => item.generics,

        Node::ImplItem(item) => item.generics,

        Node::Item(item) => match item.kind {
            ItemKind::Impl(impl_) => {
                if impl_.defaultness.is_default() {
                    is_default_impl_trait =
                        tcx.impl_trait_ref(def_id).map(|t| ty::Binder::dummy(t.subst_identity()));
                }
                impl_.generics
            }
            ItemKind::Fn(.., generics, _)
            | ItemKind::TyAlias(_, generics)
            | ItemKind::Enum(_, generics)
            | ItemKind::Struct(_, generics)
            | ItemKind::Union(_, generics) => generics,

            ItemKind::Trait(_, _, generics, self_bounds, ..)
            | ItemKind::TraitAlias(generics, self_bounds) => {
                is_trait = Some(self_bounds);
                generics
            }
            ItemKind::OpaqueTy(OpaqueTy { generics, .. }) => generics,
            _ => NO_GENERICS,
        },

        Node::ForeignItem(item) => match item.kind {
            ForeignItemKind::Static(..) => NO_GENERICS,
            ForeignItemKind::Fn(_, _, generics) => generics,
            ForeignItemKind::Type => NO_GENERICS,
        },

        _ => NO_GENERICS,
    };

    let generics = tcx.generics_of(def_id);
    let parent_count = generics.parent_count as u32;
    let has_own_self = generics.has_self && parent_count == 0;

    // Below we'll consider the bounds on the type parameters (including `Self`)
    // and the explicit where-clauses, but to get the full set of predicates
    // on a trait we must also consider the bounds that follow the trait's name,
    // like `trait Foo: A + B + C`.
    if let Some(self_bounds) = is_trait {
        predicates.extend(
            icx.astconv()
                .compute_bounds(tcx.types.self_param, self_bounds, OnlySelfBounds(false))
                .predicates(),
        );
    }

    // In default impls, we can assume that the self type implements
    // the trait. So in:
    //
    //     default impl Foo for Bar { .. }
    //
    // we add a default where clause `Foo: Bar`. We do a similar thing for traits
    // (see below). Recall that a default impl is not itself an impl, but rather a
    // set of defaults that can be incorporated into another impl.
    if let Some(trait_ref) = is_default_impl_trait {
        predicates.insert((trait_ref.without_const().to_predicate(tcx), tcx.def_span(def_id)));
    }

    // Collect the region predicates that were declared inline as
    // well. In the case of parameters declared on a fn or method, we
    // have to be careful to only iterate over early-bound regions.
    let mut index = parent_count
        + has_own_self as u32
        + super::early_bound_lifetimes_from_generics(tcx, ast_generics).count() as u32;

    trace!(?predicates);
    trace!(?ast_generics);
    trace!(?generics);

    // Collect the predicates that were written inline by the user on each
    // type parameter (e.g., `<T: Foo>`). Also add `ConstArgHasType` predicates
    // for each const parameter.
    for param in ast_generics.params {
        match param.kind {
            // We already dealt with early bound lifetimes above.
            GenericParamKind::Lifetime { .. } => (),
            GenericParamKind::Type { .. } => {
                let name = param.name.ident().name;
                let param_ty = ty::ParamTy::new(index, name).to_ty(tcx);
                index += 1;

                let mut bounds = Bounds::default();
                // Params are implicitly sized unless a `?Sized` bound is found
                icx.astconv().add_implicitly_sized(
                    &mut bounds,
                    param_ty,
                    &[],
                    Some((param.def_id, ast_generics.predicates)),
                    param.span,
                );
                trace!(?bounds);
                predicates.extend(bounds.predicates());
                trace!(?predicates);
            }
            GenericParamKind::Const { .. } => {
                let name = param.name.ident().name;
                let param_const = ty::ParamConst::new(index, name);

                let ct_ty = tcx.type_of(param.def_id.to_def_id()).subst_identity();

                let ct = tcx.mk_const(param_const, ct_ty);

                let predicate = ty::Binder::dummy(ty::PredicateKind::Clause(
                    ty::Clause::ConstArgHasType(ct, ct_ty),
                ))
                .to_predicate(tcx);
                predicates.insert((predicate, param.span));

                index += 1;
            }
        }
    }

    trace!(?predicates);
    // Add in the bounds that appear in the where-clause.
    for predicate in ast_generics.predicates {
        match predicate {
            hir::WherePredicate::BoundPredicate(bound_pred) => {
                let ty = icx.to_ty(bound_pred.bounded_ty);
                let bound_vars = icx.tcx.late_bound_vars(bound_pred.hir_id);

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
                            ty::PredicateKind::WellFormed(ty.into()),
                            bound_vars,
                        );
                        predicates.insert((predicate.to_predicate(tcx), span));
                    }
                }

                let mut bounds = Bounds::default();
                icx.astconv().add_bounds(
                    ty,
                    bound_pred.bounds.iter(),
                    &mut bounds,
                    bound_vars,
                    OnlySelfBounds(false),
                );
                predicates.extend(bounds.predicates());
            }

            hir::WherePredicate::RegionPredicate(region_pred) => {
                let r1 = icx.astconv().ast_region_to_region(&region_pred.lifetime, None);
                predicates.extend(region_pred.bounds.iter().map(|bound| {
                    let (r2, span) = match bound {
                        hir::GenericBound::Outlives(lt) => {
                            (icx.astconv().ast_region_to_region(lt, None), lt.ident.span)
                        }
                        _ => bug!(),
                    };
                    let pred = ty::Binder::dummy(ty::PredicateKind::Clause(
                        ty::Clause::RegionOutlives(ty::OutlivesPredicate(r1, r2)),
                    ))
                    .to_predicate(icx.tcx);

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
        let self_ty = tcx.type_of(def_id).subst_identity();
        let trait_ref = tcx.impl_trait_ref(def_id).map(ty::EarlyBinder::subst_identity);
        cgp::setup_constraining_predicates(
            tcx,
            &mut predicates,
            trait_ref,
            &mut cgp::parameters_for_impl(self_ty, trait_ref),
        );
    }

    // Opaque types duplicate some of their generic parameters.
    // We create bi-directional Outlives predicates between the original
    // and the duplicated parameter, to ensure that they do not get out of sync.
    if let Node::Item(&Item { kind: ItemKind::OpaqueTy(..), .. }) = node {
        let opaque_ty_id = tcx.hir().parent_id(hir_id);
        let opaque_ty_node = tcx.hir().get(opaque_ty_id);
        let Node::Ty(&Ty { kind: TyKind::OpaqueDef(_, lifetimes, _), .. }) = opaque_ty_node else {
            bug!("unexpected {opaque_ty_node:?}")
        };
        debug!(?lifetimes);
        for (arg, duplicate) in std::iter::zip(lifetimes, ast_generics.params) {
            let hir::GenericArg::Lifetime(arg) = arg else { bug!() };
            let orig_region = icx.astconv().ast_region_to_region(&arg, None);
            if !matches!(orig_region.kind(), ty::ReEarlyBound(..)) {
                // Only early-bound regions can point to the original generic parameter.
                continue;
            }

            let hir::GenericParamKind::Lifetime { .. } = duplicate.kind else { continue };
            let dup_def = duplicate.def_id.to_def_id();

            let Some(dup_index) = generics.param_def_id_to_index(tcx, dup_def) else { bug!() };

            let dup_region = tcx.mk_re_early_bound(ty::EarlyBoundRegion {
                def_id: dup_def,
                index: dup_index,
                name: duplicate.name.ident().name,
            });
            predicates.push((
                ty::Binder::dummy(ty::PredicateKind::Clause(ty::Clause::RegionOutlives(
                    ty::OutlivesPredicate(orig_region, dup_region),
                )))
                .to_predicate(icx.tcx),
                duplicate.span,
            ));
            predicates.push((
                ty::Binder::dummy(ty::PredicateKind::Clause(ty::Clause::RegionOutlives(
                    ty::OutlivesPredicate(dup_region, orig_region),
                )))
                .to_predicate(icx.tcx),
                duplicate.span,
            ));
        }
        debug!(?predicates);
    }

    ty::GenericPredicates {
        parent: generics.parent,
        predicates: tcx.arena.alloc_from_iter(predicates),
    }
}

fn const_evaluatable_predicates_of(
    tcx: TyCtxt<'_>,
    def_id: LocalDefId,
) -> FxIndexSet<(ty::Predicate<'_>, Span)> {
    struct ConstCollector<'tcx> {
        tcx: TyCtxt<'tcx>,
        preds: FxIndexSet<(ty::Predicate<'tcx>, Span)>,
    }

    impl<'tcx> intravisit::Visitor<'tcx> for ConstCollector<'tcx> {
        fn visit_anon_const(&mut self, c: &'tcx hir::AnonConst) {
            let ct = ty::Const::from_anon_const(self.tcx, c.def_id);
            if let ty::ConstKind::Unevaluated(_) = ct.kind() {
                let span = self.tcx.def_span(c.def_id);
                self.preds.insert((
                    ty::Binder::dummy(ty::PredicateKind::ConstEvaluatable(ct))
                        .to_predicate(self.tcx),
                    span,
                ));
            }
        }

        fn visit_const_param_default(&mut self, _param: HirId, _ct: &'tcx hir::AnonConst) {
            // Do not look into const param defaults,
            // these get checked when they are actually instantiated.
            //
            // We do not want the following to error:
            //
            //     struct Foo<const N: usize, const M: usize = { N + 1 }>;
            //     struct Bar<const N: usize>(Foo<N, 3>);
        }
    }

    let hir_id = tcx.hir().local_def_id_to_hir_id(def_id);
    let node = tcx.hir().get(hir_id);

    let mut collector = ConstCollector { tcx, preds: FxIndexSet::default() };
    if let hir::Node::Item(item) = node && let hir::ItemKind::Impl(impl_) = item.kind {
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
        let trait_identity_substs = InternalSubsts::identity_for_item(tcx, def_id);

        let is_assoc_item_ty = |ty: Ty<'tcx>| {
            // For a predicate from a where clause to become a bound on an
            // associated type:
            // * It must use the identity substs of the item.
            //   * We're in the scope of the trait, so we can't name any
            //     parameters of the GAT. That means that all we need to
            //     check are that the substs of the projection are the
            //     identity substs of the trait.
            // * It must be an associated type for this trait (*not* a
            //   supertrait).
            if let ty::Alias(ty::Projection, projection) = ty.kind() {
                projection.substs == trait_identity_substs
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
                ty::PredicateKind::Clause(ty::Clause::Trait(tr)) => !is_assoc_item_ty(tr.self_ty()),
                ty::PredicateKind::Clause(ty::Clause::Projection(proj)) => {
                    !is_assoc_item_ty(proj.projection_ty.self_ty())
                }
                ty::PredicateKind::Clause(ty::Clause::TypeOutlives(outlives)) => {
                    !is_assoc_item_ty(outlives.0)
                }
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
        if matches!(def_kind, DefKind::AnonConst) && tcx.lazy_normalization() {
            let hir_id = tcx.hir().local_def_id_to_hir_id(def_id);
            let parent_def_id = tcx.hir().get_parent_item(hir_id);

            if let Some(defaulted_param_def_id) =
                tcx.hir().opt_const_param_default_param_def_id(hir_id)
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
                let parent_preds = tcx.explicit_predicates_of(parent_def_id);

                // If we dont filter out `ConstArgHasType` predicates then every single defaulted const parameter
                // will ICE because of #106994. FIXME(generic_const_exprs): remove this when a more general solution
                // to #106994 is implemented.
                let filtered_predicates = parent_preds
                    .predicates
                    .into_iter()
                    .filter(|(pred, _)| {
                        if let ty::PredicateKind::Clause(ty::Clause::ConstArgHasType(ct, _)) =
                            pred.kind().skip_binder()
                        {
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

            let parent_def_kind = tcx.def_kind(parent_def_id);
            if matches!(parent_def_kind, DefKind::OpaqueTy) {
                // In `instantiate_identity` we inherit the predicates of our parent.
                // However, opaque types do not have a parent (see `gather_explicit_predicates_of`), which means
                // that we lose out on the predicates of our actual parent if we dont return those predicates here.
                //
                //
                // fn foo<T: Trait>() -> impl Iterator<Output = Another<{ <T as Trait>::ASSOC }> > { todo!() }
                //                                                        ^^^^^^^^^^^^^^^^^^^ the def id we are calling
                //                                                                            explicit_predicates_of on
                //
                // In the above code we want the anon const to have predicates in its param env for `T: Trait`.
                // However, the anon const cannot inherit predicates from its parent since it's opaque.
                //
                // To fix this, we call `explicit_predicates_of` directly on `foo`, the parent's parent.

                // In the above example this is `foo::{opaque#0}` or `impl Iterator`
                let parent_hir_id = tcx.hir().local_def_id_to_hir_id(parent_def_id.def_id);

                // In the above example this is the function `foo`
                let item_def_id = tcx.hir().get_parent_item(parent_hir_id);

                // In the above code example we would be calling `explicit_predicates_of(foo)` here
                return tcx.explicit_predicates_of(item_def_id);
            }
        }
        gather_explicit_predicates_of(tcx, def_id)
    }
}

#[derive(Copy, Clone, Debug)]
pub enum PredicateFilter {
    /// All predicates may be implied by the trait
    All,

    /// Only traits that reference `Self: ..` are implied by the trait
    SelfOnly,

    /// Only traits that reference `Self: ..` and define an associated type
    /// with the given ident are implied by the trait
    SelfThatDefines(Ident),
}

/// Ensures that the super-predicates of the trait with a `DefId`
/// of `trait_def_id` are converted and stored. This also ensures that
/// the transitive super-predicates are converted.
pub(super) fn super_predicates_of(
    tcx: TyCtxt<'_>,
    trait_def_id: LocalDefId,
) -> ty::GenericPredicates<'_> {
    implied_predicates_with_filter(tcx, trait_def_id.to_def_id(), PredicateFilter::SelfOnly)
}

pub(super) fn super_predicates_that_define_assoc_item(
    tcx: TyCtxt<'_>,
    (trait_def_id, assoc_name): (DefId, Ident),
) -> ty::GenericPredicates<'_> {
    implied_predicates_with_filter(tcx, trait_def_id, PredicateFilter::SelfThatDefines(assoc_name))
}

pub(super) fn implied_predicates_of(
    tcx: TyCtxt<'_>,
    trait_def_id: LocalDefId,
) -> ty::GenericPredicates<'_> {
    if tcx.is_trait_alias(trait_def_id.to_def_id()) {
        implied_predicates_with_filter(tcx, trait_def_id.to_def_id(), PredicateFilter::All)
    } else {
        tcx.super_predicates_of(trait_def_id)
    }
}

/// Ensures that the super-predicates of the trait with a `DefId`
/// of `trait_def_id` are converted and stored. This also ensures that
/// the transitive super-predicates are converted.
pub(super) fn implied_predicates_with_filter(
    tcx: TyCtxt<'_>,
    trait_def_id: DefId,
    filter: PredicateFilter,
) -> ty::GenericPredicates<'_> {
    let Some(trait_def_id) = trait_def_id.as_local() else {
        // if `assoc_name` is None, then the query should've been redirected to an
        // external provider
        assert!(matches!(filter, PredicateFilter::SelfThatDefines(_)));
        return tcx.super_predicates_of(trait_def_id);
    };

    let trait_hir_id = tcx.hir().local_def_id_to_hir_id(trait_def_id);

    let Node::Item(item) = tcx.hir().get(trait_hir_id) else {
        bug!("trait_node_id {} is not an item", trait_hir_id);
    };

    let (generics, bounds) = match item.kind {
        hir::ItemKind::Trait(.., generics, supertraits, _) => (generics, supertraits),
        hir::ItemKind::TraitAlias(generics, supertraits) => (generics, supertraits),
        _ => span_bug!(item.span, "super_predicates invoked on non-trait"),
    };

    let icx = ItemCtxt::new(tcx, trait_def_id);

    let self_param_ty = tcx.types.self_param;
    let (superbounds, where_bounds_that_match) = match filter {
        PredicateFilter::All => (
            // Convert the bounds that follow the colon (or equal in trait aliases)
            icx.astconv().compute_bounds(self_param_ty, bounds, OnlySelfBounds(false)),
            // Also include all where clause bounds
            icx.type_parameter_bounds_in_generics(
                generics,
                item.owner_id.def_id,
                self_param_ty,
                OnlySelfBounds(false),
                None,
            ),
        ),
        PredicateFilter::SelfOnly => (
            // Convert the bounds that follow the colon (or equal in trait aliases)
            icx.astconv().compute_bounds(self_param_ty, bounds, OnlySelfBounds(true)),
            // Include where clause bounds for `Self`
            icx.type_parameter_bounds_in_generics(
                generics,
                item.owner_id.def_id,
                self_param_ty,
                OnlySelfBounds(true),
                None,
            ),
        ),
        PredicateFilter::SelfThatDefines(assoc_name) => (
            // Convert the bounds that follow the colon (or equal) that reference the associated name
            icx.astconv().compute_bounds_that_match_assoc_item(self_param_ty, bounds, assoc_name),
            // Include where clause bounds for `Self` that reference the associated name
            icx.type_parameter_bounds_in_generics(
                generics,
                item.owner_id.def_id,
                self_param_ty,
                OnlySelfBounds(true),
                Some(assoc_name),
            ),
        ),
    };

    // Combine the two lists to form the complete set of superbounds:
    let implied_bounds =
        &*tcx.arena.alloc_from_iter(superbounds.predicates().chain(where_bounds_that_match));
    debug!(?implied_bounds);

    // Now require that immediate supertraits are converted, which will, in
    // turn, reach indirect supertraits, so we detect cycles now instead of
    // overflowing during elaboration.
    if matches!(filter, PredicateFilter::SelfOnly) {
        for &(pred, span) in implied_bounds {
            debug!("superbound: {:?}", pred);
            if let ty::PredicateKind::Clause(ty::Clause::Trait(bound)) = pred.kind().skip_binder()
                && bound.polarity == ty::ImplPolarity::Positive
            {
                tcx.at(span).super_predicates_of(bound.def_id());
            }
        }
    }

    ty::GenericPredicates { parent: None, predicates: implied_bounds }
}

/// Returns the predicates defined on `item_def_id` of the form
/// `X: Foo` where `X` is the type parameter `def_id`.
#[instrument(level = "trace", skip(tcx))]
pub(super) fn type_param_predicates(
    tcx: TyCtxt<'_>,
    (item_def_id, def_id, assoc_name): (LocalDefId, LocalDefId, Ident),
) -> ty::GenericPredicates<'_> {
    use rustc_hir::*;

    // In the AST, bounds can derive from two places. Either
    // written inline like `<T: Foo>` or in a where-clause like
    // `where T: Foo`.

    let param_id = tcx.hir().local_def_id_to_hir_id(def_id);
    let param_owner = tcx.hir().ty_param_owner(def_id);
    let generics = tcx.generics_of(param_owner);
    let index = generics.param_def_id_to_index[&def_id.to_def_id()];
    let ty = tcx.mk_ty_param(index, tcx.hir().ty_param_name(def_id));

    // Don't look for bounds where the type parameter isn't in scope.
    let parent = if item_def_id == param_owner {
        None
    } else {
        tcx.generics_of(item_def_id).parent.map(|def_id| def_id.expect_local())
    };

    let mut result = parent
        .map(|parent| {
            let icx = ItemCtxt::new(tcx, parent);
            icx.get_type_parameter_bounds(DUMMY_SP, def_id, assoc_name)
        })
        .unwrap_or_default();
    let mut extend = None;

    let item_hir_id = tcx.hir().local_def_id_to_hir_id(item_def_id);
    let ast_generics = match tcx.hir().get(item_hir_id) {
        Node::TraitItem(item) => &item.generics,

        Node::ImplItem(item) => &item.generics,

        Node::Item(item) => {
            match item.kind {
                ItemKind::Fn(.., generics, _)
                | ItemKind::Impl(&hir::Impl { generics, .. })
                | ItemKind::TyAlias(_, generics)
                | ItemKind::OpaqueTy(OpaqueTy {
                    generics,
                    origin: hir::OpaqueTyOrigin::TyAlias,
                    ..
                })
                | ItemKind::Enum(_, generics)
                | ItemKind::Struct(_, generics)
                | ItemKind::Union(_, generics) => generics,
                ItemKind::Trait(_, _, generics, ..) => {
                    // Implied `Self: Trait` and supertrait bounds.
                    if param_id == item_hir_id {
                        let identity_trait_ref =
                            ty::TraitRef::identity(tcx, item_def_id.to_def_id());
                        extend =
                            Some((identity_trait_ref.without_const().to_predicate(tcx), item.span));
                    }
                    generics
                }
                _ => return result,
            }
        }

        Node::ForeignItem(item) => match item.kind {
            ForeignItemKind::Fn(_, _, generics) => generics,
            _ => return result,
        },

        _ => return result,
    };

    let icx = ItemCtxt::new(tcx, item_def_id);
    let extra_predicates = extend.into_iter().chain(
        icx.type_parameter_bounds_in_generics(
            ast_generics,
            def_id,
            ty,
            OnlySelfBounds(true),
            Some(assoc_name),
        )
        .into_iter()
        .filter(|(predicate, _)| match predicate.kind().skip_binder() {
            ty::PredicateKind::Clause(ty::Clause::Trait(data)) => data.self_ty().is_param(index),
            _ => false,
        }),
    );
    result.predicates =
        tcx.arena.alloc_from_iter(result.predicates.iter().copied().chain(extra_predicates));
    result
}

impl<'tcx> ItemCtxt<'tcx> {
    /// Finds bounds from `hir::Generics`. This requires scanning through the
    /// AST. We do this to avoid having to convert *all* the bounds, which
    /// would create artificial cycles. Instead, we can only convert the
    /// bounds for a type parameter `X` if `X::Foo` is used.
    #[instrument(level = "trace", skip(self, ast_generics))]
    fn type_parameter_bounds_in_generics(
        &self,
        ast_generics: &'tcx hir::Generics<'tcx>,
        param_def_id: LocalDefId,
        ty: Ty<'tcx>,
        only_self_bounds: OnlySelfBounds,
        assoc_name: Option<Ident>,
    ) -> Vec<(ty::Predicate<'tcx>, Span)> {
        let mut bounds = Bounds::default();

        for predicate in ast_generics.predicates {
            let hir::WherePredicate::BoundPredicate(predicate) = predicate else {
                continue;
            };

            let bound_ty = if predicate.is_param_bound(param_def_id.to_def_id()) {
                ty
            } else if !only_self_bounds.0 {
                self.to_ty(predicate.bounded_ty)
            } else {
                continue;
            };

            let bound_vars = self.tcx.late_bound_vars(predicate.hir_id);
            self.astconv().add_bounds(
                bound_ty,
                predicate.bounds.iter().filter(|bound| {
                    assoc_name
                        .map_or(true, |assoc_name| self.bound_defines_assoc_item(bound, assoc_name))
                }),
                &mut bounds,
                bound_vars,
                only_self_bounds,
            );
        }

        bounds.predicates().collect()
    }

    #[instrument(level = "trace", skip(self))]
    fn bound_defines_assoc_item(&self, b: &hir::GenericBound<'_>, assoc_name: Ident) -> bool {
        match b {
            hir::GenericBound::Trait(poly_trait_ref, _) => {
                let trait_ref = &poly_trait_ref.trait_ref;
                if let Some(trait_did) = trait_ref.trait_def_id() {
                    self.tcx.trait_may_define_assoc_item(trait_did, assoc_name)
                } else {
                    false
                }
            }
            _ => false,
        }
    }
}
