use crate::astconv::AstConv;
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
use rustc_middle::ty::ToPredicate;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_span::symbol::{sym, Ident};
use rustc_span::{Span, DUMMY_SP};

#[derive(Debug)]
struct OnlySelfBounds(bool);

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
fn gather_explicit_predicates_of(tcx: TyCtxt<'_>, def_id: DefId) -> ty::GenericPredicates<'_> {
    use rustc_hir::*;

    let hir_id = tcx.hir().local_def_id_to_hir_id(def_id.expect_local());
    let node = tcx.hir().get(hir_id);

    let mut is_trait = None;
    let mut is_default_impl_trait = None;

    let icx = ItemCtxt::new(tcx, def_id);

    const NO_GENERICS: &hir::Generics<'_> = hir::Generics::empty();

    // We use an `IndexSet` to preserves order of insertion.
    // Preserving the order of insertion is important here so as not to break UI tests.
    let mut predicates: FxIndexSet<(ty::Predicate<'_>, Span)> = FxIndexSet::default();

    let ast_generics = match node {
        Node::TraitItem(item) => item.generics,

        Node::ImplItem(item) => item.generics,

        Node::Item(item) => {
            match item.kind {
                ItemKind::Impl(ref impl_) => {
                    if impl_.defaultness.is_default() {
                        is_default_impl_trait = tcx.impl_trait_ref(def_id).map(ty::Binder::dummy);
                    }
                    &impl_.generics
                }
                ItemKind::Fn(.., ref generics, _)
                | ItemKind::TyAlias(_, ref generics)
                | ItemKind::Enum(_, ref generics)
                | ItemKind::Struct(_, ref generics)
                | ItemKind::Union(_, ref generics) => *generics,

                ItemKind::Trait(_, _, ref generics, ..) => {
                    is_trait = Some(ty::TraitRef::identity(tcx, def_id));
                    *generics
                }
                ItemKind::TraitAlias(ref generics, _) => {
                    is_trait = Some(ty::TraitRef::identity(tcx, def_id));
                    *generics
                }
                ItemKind::OpaqueTy(OpaqueTy {
                    origin: hir::OpaqueTyOrigin::AsyncFn(..) | hir::OpaqueTyOrigin::FnReturn(..),
                    ..
                }) => {
                    // return-position impl trait
                    //
                    // We don't inherit predicates from the parent here:
                    // If we have, say `fn f<'a, T: 'a>() -> impl Sized {}`
                    // then the return type is `f::<'static, T>::{{opaque}}`.
                    //
                    // If we inherited the predicates of `f` then we would
                    // require that `T: 'static` to show that the return
                    // type is well-formed.
                    //
                    // The only way to have something with this opaque type
                    // is from the return type of the containing function,
                    // which will ensure that the function's predicates
                    // hold.
                    return ty::GenericPredicates { parent: None, predicates: &[] };
                }
                ItemKind::OpaqueTy(OpaqueTy {
                    ref generics,
                    origin: hir::OpaqueTyOrigin::TyAlias,
                    ..
                }) => {
                    // type-alias impl trait
                    generics
                }

                _ => NO_GENERICS,
            }
        }

        Node::ForeignItem(item) => match item.kind {
            ForeignItemKind::Static(..) => NO_GENERICS,
            ForeignItemKind::Fn(_, _, ref generics) => *generics,
            ForeignItemKind::Type => NO_GENERICS,
        },

        _ => NO_GENERICS,
    };

    let generics = tcx.generics_of(def_id);
    let parent_count = generics.parent_count as u32;
    let has_own_self = generics.has_self && parent_count == 0;

    // Below we'll consider the bounds on the type parameters (including `Self`)
    // and the explicit where-clauses, but to get the full set of predicates
    // on a trait we need to add in the supertrait bounds and bounds found on
    // associated types.
    if let Some(_trait_ref) = is_trait {
        predicates.extend(tcx.super_predicates_of(def_id).predicates.iter().cloned());
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

    // Collect the predicates that were written inline by the user on each
    // type parameter (e.g., `<T: Foo>`).
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
                <dyn AstConv<'_>>::add_implicitly_sized(
                    &icx,
                    &mut bounds,
                    &[],
                    Some((param.hir_id, ast_generics.predicates)),
                    param.span,
                );
                trace!(?bounds);
                predicates.extend(bounds.predicates(tcx, param_ty));
                trace!(?predicates);
            }
            GenericParamKind::Const { .. } => {
                // Bounds on const parameters are currently not possible.
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
                <dyn AstConv<'_>>::add_bounds(
                    &icx,
                    ty,
                    bound_pred.bounds.iter(),
                    &mut bounds,
                    bound_vars,
                );
                predicates.extend(bounds.predicates(tcx, ty));
            }

            hir::WherePredicate::RegionPredicate(region_pred) => {
                let r1 = <dyn AstConv<'_>>::ast_region_to_region(&icx, &region_pred.lifetime, None);
                predicates.extend(region_pred.bounds.iter().map(|bound| {
                    let (r2, span) = match bound {
                        hir::GenericBound::Outlives(lt) => {
                            (<dyn AstConv<'_>>::ast_region_to_region(&icx, lt, None), lt.span)
                        }
                        _ => bug!(),
                    };
                    let pred = ty::Binder::dummy(ty::PredicateKind::RegionOutlives(
                        ty::OutlivesPredicate(r1, r2),
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
        predicates.extend(const_evaluatable_predicates_of(tcx, def_id.expect_local()));
    }

    let mut predicates: Vec<_> = predicates.into_iter().collect();

    // Subtle: before we store the predicates into the tcx, we
    // sort them so that predicates like `T: Foo<Item=U>` come
    // before uses of `U`.  This avoids false ambiguity errors
    // in trait checking. See `setup_constraining_predicates`
    // for details.
    if let Node::Item(&Item { kind: ItemKind::Impl { .. }, .. }) = node {
        let self_ty = tcx.type_of(def_id);
        let trait_ref = tcx.impl_trait_ref(def_id);
        cgp::setup_constraining_predicates(
            tcx,
            &mut predicates,
            trait_ref,
            &mut cgp::parameters_for_impl(self_ty, trait_ref),
        );
    }

    ty::GenericPredicates {
        parent: generics.parent,
        predicates: tcx.arena.alloc_from_iter(predicates),
    }
}

fn const_evaluatable_predicates_of<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: LocalDefId,
) -> FxIndexSet<(ty::Predicate<'tcx>, Span)> {
    struct ConstCollector<'tcx> {
        tcx: TyCtxt<'tcx>,
        preds: FxIndexSet<(ty::Predicate<'tcx>, Span)>,
    }

    impl<'tcx> intravisit::Visitor<'tcx> for ConstCollector<'tcx> {
        fn visit_anon_const(&mut self, c: &'tcx hir::AnonConst) {
            let def_id = self.tcx.hir().local_def_id(c.hir_id);
            let ct = ty::Const::from_anon_const(self.tcx, def_id);
            if let ty::ConstKind::Unevaluated(uv) = ct.kind() {
                let span = self.tcx.hir().span(c.hir_id);
                self.preds.insert((
                    ty::Binder::dummy(ty::PredicateKind::ConstEvaluatable(uv))
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
    if let hir::Node::Item(item) = node && let hir::ItemKind::Impl(ref impl_) = item.kind {
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
    gather_explicit_predicates_of(tcx, def_id.to_def_id())
}

pub(super) fn explicit_predicates_of<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: DefId,
) -> ty::GenericPredicates<'tcx> {
    let def_kind = tcx.def_kind(def_id);
    if let DefKind::Trait = def_kind {
        // Remove bounds on associated types from the predicates, they will be
        // returned by `explicit_item_bounds`.
        let predicates_and_bounds = tcx.trait_explicit_predicates_and_bounds(def_id.expect_local());
        let trait_identity_substs = InternalSubsts::identity_for_item(tcx, def_id);

        let is_assoc_item_ty = |ty: Ty<'tcx>| {
            // For a predicate from a where clause to become a bound on an
            // associated type:
            // * It must use the identity substs of the item.
            //     * Since any generic parameters on the item are not in scope,
            //       this means that the item is not a GAT, and its identity
            //       substs are the same as the trait's.
            // * It must be an associated type for this trait (*not* a
            //   supertrait).
            if let ty::Projection(projection) = ty.kind() {
                projection.substs == trait_identity_substs
                    && tcx.associated_item(projection.item_def_id).container_id(tcx) == def_id
            } else {
                false
            }
        };

        let predicates: Vec<_> = predicates_and_bounds
            .predicates
            .iter()
            .copied()
            .filter(|(pred, _)| match pred.kind().skip_binder() {
                ty::PredicateKind::Trait(tr) => !is_assoc_item_ty(tr.self_ty()),
                ty::PredicateKind::Projection(proj) => {
                    !is_assoc_item_ty(proj.projection_ty.self_ty())
                }
                ty::PredicateKind::TypeOutlives(outlives) => !is_assoc_item_ty(outlives.0),
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
            let hir_id = tcx.hir().local_def_id_to_hir_id(def_id.expect_local());
            if tcx.hir().opt_const_param_default_param_hir_id(hir_id).is_some() {
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
                let item_def_id = tcx.hir().get_parent_item(hir_id);
                // In the above code example we would be calling `explicit_predicates_of(Foo)` here
                return tcx.explicit_predicates_of(item_def_id);
            }
        }
        gather_explicit_predicates_of(tcx, def_id)
    }
}

/// Ensures that the super-predicates of the trait with a `DefId`
/// of `trait_def_id` are converted and stored. This also ensures that
/// the transitive super-predicates are converted.
pub(super) fn super_predicates_of(
    tcx: TyCtxt<'_>,
    trait_def_id: DefId,
) -> ty::GenericPredicates<'_> {
    tcx.super_predicates_that_define_assoc_type((trait_def_id, None))
}

/// Ensures that the super-predicates of the trait with a `DefId`
/// of `trait_def_id` are converted and stored. This also ensures that
/// the transitive super-predicates are converted.
pub(super) fn super_predicates_that_define_assoc_type(
    tcx: TyCtxt<'_>,
    (trait_def_id, assoc_name): (DefId, Option<Ident>),
) -> ty::GenericPredicates<'_> {
    if trait_def_id.is_local() {
        debug!("local trait");
        let trait_hir_id = tcx.hir().local_def_id_to_hir_id(trait_def_id.expect_local());

        let Node::Item(item) = tcx.hir().get(trait_hir_id) else {
            bug!("trait_node_id {} is not an item", trait_hir_id);
        };

        let (generics, bounds) = match item.kind {
            hir::ItemKind::Trait(.., ref generics, ref supertraits, _) => (generics, supertraits),
            hir::ItemKind::TraitAlias(ref generics, ref supertraits) => (generics, supertraits),
            _ => span_bug!(item.span, "super_predicates invoked on non-trait"),
        };

        let icx = ItemCtxt::new(tcx, trait_def_id);

        // Convert the bounds that follow the colon, e.g., `Bar + Zed` in `trait Foo: Bar + Zed`.
        let self_param_ty = tcx.types.self_param;
        let superbounds1 = if let Some(assoc_name) = assoc_name {
            <dyn AstConv<'_>>::compute_bounds_that_match_assoc_type(
                &icx,
                self_param_ty,
                bounds,
                assoc_name,
            )
        } else {
            <dyn AstConv<'_>>::compute_bounds(&icx, self_param_ty, bounds)
        };

        let superbounds1 = superbounds1.predicates(tcx, self_param_ty);

        // Convert any explicit superbounds in the where-clause,
        // e.g., `trait Foo where Self: Bar`.
        // In the case of trait aliases, however, we include all bounds in the where-clause,
        // so e.g., `trait Foo = where u32: PartialEq<Self>` would include `u32: PartialEq<Self>`
        // as one of its "superpredicates".
        let is_trait_alias = tcx.is_trait_alias(trait_def_id);
        let superbounds2 = icx.type_parameter_bounds_in_generics(
            generics,
            item.hir_id(),
            self_param_ty,
            OnlySelfBounds(!is_trait_alias),
            assoc_name,
        );

        // Combine the two lists to form the complete set of superbounds:
        let superbounds = &*tcx.arena.alloc_from_iter(superbounds1.into_iter().chain(superbounds2));
        debug!(?superbounds);

        // Now require that immediate supertraits are converted,
        // which will, in turn, reach indirect supertraits.
        if assoc_name.is_none() {
            // Now require that immediate supertraits are converted,
            // which will, in turn, reach indirect supertraits.
            for &(pred, span) in superbounds {
                debug!("superbound: {:?}", pred);
                if let ty::PredicateKind::Trait(bound) = pred.kind().skip_binder() {
                    tcx.at(span).super_predicates_of(bound.def_id());
                }
            }
        }

        ty::GenericPredicates { parent: None, predicates: superbounds }
    } else {
        // if `assoc_name` is None, then the query should've been redirected to an
        // external provider
        assert!(assoc_name.is_some());
        tcx.super_predicates_of(trait_def_id)
    }
}

/// Returns the predicates defined on `item_def_id` of the form
/// `X: Foo` where `X` is the type parameter `def_id`.
#[instrument(level = "trace", skip(tcx))]
pub(super) fn type_param_predicates(
    tcx: TyCtxt<'_>,
    (item_def_id, def_id, assoc_name): (DefId, LocalDefId, Ident),
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
    let parent = if item_def_id == param_owner.to_def_id() {
        None
    } else {
        tcx.generics_of(item_def_id).parent
    };

    let mut result = parent
        .map(|parent| {
            let icx = ItemCtxt::new(tcx, parent);
            icx.get_type_parameter_bounds(DUMMY_SP, def_id.to_def_id(), assoc_name)
        })
        .unwrap_or_default();
    let mut extend = None;

    let item_hir_id = tcx.hir().local_def_id_to_hir_id(item_def_id.expect_local());
    let ast_generics = match tcx.hir().get(item_hir_id) {
        Node::TraitItem(item) => &item.generics,

        Node::ImplItem(item) => &item.generics,

        Node::Item(item) => {
            match item.kind {
                ItemKind::Fn(.., ref generics, _)
                | ItemKind::Impl(hir::Impl { ref generics, .. })
                | ItemKind::TyAlias(_, ref generics)
                | ItemKind::OpaqueTy(OpaqueTy {
                    ref generics,
                    origin: hir::OpaqueTyOrigin::TyAlias,
                    ..
                })
                | ItemKind::Enum(_, ref generics)
                | ItemKind::Struct(_, ref generics)
                | ItemKind::Union(_, ref generics) => generics,
                ItemKind::Trait(_, _, ref generics, ..) => {
                    // Implied `Self: Trait` and supertrait bounds.
                    if param_id == item_hir_id {
                        let identity_trait_ref = ty::TraitRef::identity(tcx, item_def_id);
                        extend =
                            Some((identity_trait_ref.without_const().to_predicate(tcx), item.span));
                    }
                    generics
                }
                _ => return result,
            }
        }

        Node::ForeignItem(item) => match item.kind {
            ForeignItemKind::Fn(_, _, ref generics) => generics,
            _ => return result,
        },

        _ => return result,
    };

    let icx = ItemCtxt::new(tcx, item_def_id);
    let extra_predicates = extend.into_iter().chain(
        icx.type_parameter_bounds_in_generics(
            ast_generics,
            param_id,
            ty,
            OnlySelfBounds(true),
            Some(assoc_name),
        )
        .into_iter()
        .filter(|(predicate, _)| match predicate.kind().skip_binder() {
            ty::PredicateKind::Trait(data) => data.self_ty().is_param(index),
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
        param_id: hir::HirId,
        ty: Ty<'tcx>,
        only_self_bounds: OnlySelfBounds,
        assoc_name: Option<Ident>,
    ) -> Vec<(ty::Predicate<'tcx>, Span)> {
        let param_def_id = self.tcx.hir().local_def_id(param_id).to_def_id();
        trace!(?param_def_id);
        ast_generics
            .predicates
            .iter()
            .filter_map(|wp| match *wp {
                hir::WherePredicate::BoundPredicate(ref bp) => Some(bp),
                _ => None,
            })
            .flat_map(|bp| {
                let bt = if bp.is_param_bound(param_def_id) {
                    Some(ty)
                } else if !only_self_bounds.0 {
                    Some(self.to_ty(bp.bounded_ty))
                } else {
                    None
                };
                let bvars = self.tcx.late_bound_vars(bp.hir_id);

                bp.bounds.iter().filter_map(move |b| bt.map(|bt| (bt, b, bvars))).filter(
                    |(_, b, _)| match assoc_name {
                        Some(assoc_name) => self.bound_defines_assoc_item(b, assoc_name),
                        None => true,
                    },
                )
            })
            .flat_map(|(bt, b, bvars)| predicates_from_bound(self, bt, b, bvars))
            .collect()
    }

    #[instrument(level = "trace", skip(self))]
    fn bound_defines_assoc_item(&self, b: &hir::GenericBound<'_>, assoc_name: Ident) -> bool {
        match b {
            hir::GenericBound::Trait(poly_trait_ref, _) => {
                let trait_ref = &poly_trait_ref.trait_ref;
                if let Some(trait_did) = trait_ref.trait_def_id() {
                    self.tcx.trait_may_define_assoc_type(trait_did, assoc_name)
                } else {
                    false
                }
            }
            _ => false,
        }
    }
}

/// Converts a specific `GenericBound` from the AST into a set of
/// predicates that apply to the self type. A vector is returned
/// because this can be anywhere from zero predicates (`T: ?Sized` adds no
/// predicates) to one (`T: Foo`) to many (`T: Bar<X = i32>` adds `T: Bar`
/// and `<T as Bar>::X == i32`).
fn predicates_from_bound<'tcx>(
    astconv: &dyn AstConv<'tcx>,
    param_ty: Ty<'tcx>,
    bound: &'tcx hir::GenericBound<'tcx>,
    bound_vars: &'tcx ty::List<ty::BoundVariableKind>,
) -> Vec<(ty::Predicate<'tcx>, Span)> {
    let mut bounds = Bounds::default();
    astconv.add_bounds(param_ty, [bound].into_iter(), &mut bounds, bound_vars);
    bounds.predicates(astconv.tcx(), param_ty).collect()
}
