//! # Minimal Specialization
//!
//! This module contains the checks for sound specialization used when the
//! `min_specialization` feature is enabled. This requires that the impl is
//! *always applicable*.
//!
//! If `impl1` specializes `impl2` then `impl1` is always applicable if we know
//! that all the bounds of `impl2` are satisfied, and all of the bounds of
//! `impl1` are satisfied for some choice of lifetimes then we know that
//! `impl1` applies for any choice of lifetimes.
//!
//! ## Basic approach
//!
//! To enforce this requirement on specializations we take the following
//! approach:
//!
//! 1. Match up the substs for `impl2` so that the implemented trait and
//!    self-type match those for `impl1`.
//! 2. Check for any direct use of `'static` in the substs of `impl2`.
//! 3. Check that all of the generic parameters of `impl1` occur at most once
//!    in the *unconstrained* substs for `impl2`. A parameter is constrained if
//!    its value is completely determined by an associated type projection
//!    predicate.
//! 4. Check that all predicates on `impl1` either exist on `impl2` (after
//!    matching substs), or are well-formed predicates for the trait's type
//!    arguments.
//!
//! ## Example
//!
//! Suppose we have the following always applicable impl:
//!
//! ```ignore (illustrative)
//! impl<T> SpecExtend<T> for std::vec::IntoIter<T> { /* specialized impl */ }
//! impl<T, I: Iterator<Item=T>> SpecExtend<T> for I { /* default impl */ }
//! ```
//!
//! We get that the subst for `impl2` are `[T, std::vec::IntoIter<T>]`. `T` is
//! constrained to be `<I as Iterator>::Item`, so we check only
//! `std::vec::IntoIter<T>` for repeated parameters, which it doesn't have. The
//! predicates of `impl1` are only `T: Sized`, which is also a predicate of
//! `impl2`. So this specialization is sound.
//!
//! ## Extensions
//!
//! Unfortunately not all specializations in the standard library are allowed
//! by this. So there are two extensions to these rules that allow specializing
//! on some traits: that is, using them as bounds on the specializing impl,
//! even when they don't occur in the base impl.
//!
//! ### rustc_specialization_trait
//!
//! If a trait is always applicable, then it's sound to specialize on it. We
//! check trait is always applicable in the same way as impls, except that step
//! 4 is now "all predicates on `impl1` are always applicable". We require that
//! `specialization` or `min_specialization` is enabled to implement these
//! traits.
//!
//! ### rustc_unsafe_specialization_marker
//!
//! There are also some specialization on traits with no methods, including the
//! stable `FusedIterator` trait. We allow marking marker traits with an
//! unstable attribute that means we ignore them in point 3 of the checks
//! above. This is unsound, in the sense that the specialized impl may be used
//! when it doesn't apply, but we allow it in the short term since it can't
//! cause use after frees with purely safe code in the same way as specializing
//! on traits with methods can.

use crate::constrained_generic_params as cgp;
use crate::errors::SubstsOnOverriddenImpl;

use rustc_data_structures::fx::FxHashSet;
use rustc_hir as hir;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_infer::infer::outlives::env::OutlivesEnvironment;
use rustc_infer::infer::TyCtxtInferExt;
use rustc_infer::traits::specialization_graph::Node;
use rustc_middle::ty::subst::{GenericArg, InternalSubsts, SubstsRef};
use rustc_middle::ty::trait_def::TraitSpecializationKind;
use rustc_middle::ty::{self, TyCtxt, TypeVisitableExt};
use rustc_span::Span;
use rustc_trait_selection::traits::error_reporting::TypeErrCtxtExt;
use rustc_trait_selection::traits::outlives_bounds::InferCtxtExt as _;
use rustc_trait_selection::traits::{self, translate_substs, wf, ObligationCtxt};

pub(super) fn check_min_specialization(tcx: TyCtxt<'_>, impl_def_id: LocalDefId) {
    if let Some(node) = parent_specialization_node(tcx, impl_def_id) {
        check_always_applicable(tcx, impl_def_id, node);
    }
}

fn parent_specialization_node(tcx: TyCtxt<'_>, impl1_def_id: LocalDefId) -> Option<Node> {
    let trait_ref = tcx.impl_trait_ref(impl1_def_id)?;
    let trait_def = tcx.trait_def(trait_ref.skip_binder().def_id);

    let impl2_node = trait_def.ancestors(tcx, impl1_def_id.to_def_id()).ok()?.nth(1)?;

    let always_applicable_trait =
        matches!(trait_def.specialization_kind, TraitSpecializationKind::AlwaysApplicable);
    if impl2_node.is_from_trait() && !always_applicable_trait {
        // Implementing a normal trait isn't a specialization.
        return None;
    }
    Some(impl2_node)
}

/// Check that `impl1` is a sound specialization
#[instrument(level = "debug", skip(tcx))]
fn check_always_applicable(tcx: TyCtxt<'_>, impl1_def_id: LocalDefId, impl2_node: Node) {
    if let Some((impl1_substs, impl2_substs)) = get_impl_substs(tcx, impl1_def_id, impl2_node) {
        let impl2_def_id = impl2_node.def_id();
        debug!(?impl2_def_id, ?impl2_substs);

        let parent_substs = if impl2_node.is_from_trait() {
            impl2_substs.to_vec()
        } else {
            unconstrained_parent_impl_substs(tcx, impl2_def_id, impl2_substs)
        };

        let span = tcx.def_span(impl1_def_id);
        check_constness(tcx, impl1_def_id, impl2_node, span);
        check_static_lifetimes(tcx, &parent_substs, span);
        check_duplicate_params(tcx, impl1_substs, &parent_substs, span);
        check_predicates(tcx, impl1_def_id, impl1_substs, impl2_node, impl2_substs, span);
    }
}

/// Check that the specializing impl `impl1` is at least as const as the base
/// impl `impl2`
fn check_constness(tcx: TyCtxt<'_>, impl1_def_id: LocalDefId, impl2_node: Node, span: Span) {
    if impl2_node.is_from_trait() {
        // This isn't a specialization
        return;
    }

    let impl1_constness = tcx.constness(impl1_def_id.to_def_id());
    let impl2_constness = tcx.constness(impl2_node.def_id());

    if let hir::Constness::Const = impl2_constness {
        if let hir::Constness::NotConst = impl1_constness {
            tcx.sess
                .struct_span_err(span, "cannot specialize on const impl with non-const impl")
                .emit();
        }
    }
}

/// Given a specializing impl `impl1`, and the base impl `impl2`, returns two
/// substitutions `(S1, S2)` that equate their trait references. The returned
/// types are expressed in terms of the generics of `impl1`.
///
/// Example
///
/// ```ignore (illustrative)
/// impl<A, B> Foo<A> for B { /* impl2 */ }
/// impl<C> Foo<Vec<C>> for C { /* impl1 */ }
/// ```
///
/// Would return `S1 = [C]` and `S2 = [Vec<C>, C]`.
fn get_impl_substs(
    tcx: TyCtxt<'_>,
    impl1_def_id: LocalDefId,
    impl2_node: Node,
) -> Option<(SubstsRef<'_>, SubstsRef<'_>)> {
    let infcx = &tcx.infer_ctxt().build();
    let ocx = ObligationCtxt::new(infcx);
    let param_env = tcx.param_env(impl1_def_id);

    let assumed_wf_types =
        ocx.assumed_wf_types(param_env, tcx.def_span(impl1_def_id), impl1_def_id);

    let impl1_substs = InternalSubsts::identity_for_item(tcx, impl1_def_id.to_def_id());
    let impl2_substs =
        translate_substs(infcx, param_env, impl1_def_id.to_def_id(), impl1_substs, impl2_node);

    let errors = ocx.select_all_or_error();
    if !errors.is_empty() {
        ocx.infcx.err_ctxt().report_fulfillment_errors(&errors, None);
        return None;
    }

    let implied_bounds = infcx.implied_bounds_tys(param_env, impl1_def_id, assumed_wf_types);
    let outlives_env = OutlivesEnvironment::with_bounds(param_env, Some(infcx), implied_bounds);
    let _ =
        infcx.err_ctxt().check_region_obligations_and_report_errors(impl1_def_id, &outlives_env);
    let Ok(impl2_substs) = infcx.fully_resolve(impl2_substs) else {
        let span = tcx.def_span(impl1_def_id);
        tcx.sess.emit_err(SubstsOnOverriddenImpl { span });
        return None;
    };
    Some((impl1_substs, impl2_substs))
}

/// Returns a list of all of the unconstrained subst of the given impl.
///
/// For example given the impl:
///
/// impl<'a, T, I> ... where &'a I: IntoIterator<Item=&'a T>
///
/// This would return the substs corresponding to `['a, I]`, because knowing
/// `'a` and `I` determines the value of `T`.
fn unconstrained_parent_impl_substs<'tcx>(
    tcx: TyCtxt<'tcx>,
    impl_def_id: DefId,
    impl_substs: SubstsRef<'tcx>,
) -> Vec<GenericArg<'tcx>> {
    let impl_generic_predicates = tcx.predicates_of(impl_def_id);
    let mut unconstrained_parameters = FxHashSet::default();
    let mut constrained_params = FxHashSet::default();
    let impl_trait_ref = tcx.impl_trait_ref(impl_def_id).map(ty::EarlyBinder::subst_identity);

    // Unfortunately the functions in `constrained_generic_parameters` don't do
    // what we want here. We want only a list of constrained parameters while
    // the functions in `cgp` add the constrained parameters to a list of
    // unconstrained parameters.
    for (predicate, _) in impl_generic_predicates.predicates.iter() {
        if let ty::PredicateKind::Clause(ty::Clause::Projection(proj)) =
            predicate.kind().skip_binder()
        {
            let projection_ty = proj.projection_ty;
            let projected_ty = proj.term;

            let unbound_trait_ref = projection_ty.trait_ref(tcx);
            if Some(unbound_trait_ref) == impl_trait_ref {
                continue;
            }

            unconstrained_parameters.extend(cgp::parameters_for(&projection_ty, true));

            for param in cgp::parameters_for(&projected_ty, false) {
                if !unconstrained_parameters.contains(&param) {
                    constrained_params.insert(param.0);
                }
            }

            unconstrained_parameters.extend(cgp::parameters_for(&projected_ty, true));
        }
    }

    impl_substs
        .iter()
        .enumerate()
        .filter(|&(idx, _)| !constrained_params.contains(&(idx as u32)))
        .map(|(_, arg)| arg)
        .collect()
}

/// Check that parameters of the derived impl don't occur more than once in the
/// equated substs of the base impl.
///
/// For example forbid the following:
///
/// ```ignore (illustrative)
/// impl<A> Tr for A { }
/// impl<B> Tr for (B, B) { }
/// ```
///
/// Note that only consider the unconstrained parameters of the base impl:
///
/// ```ignore (illustrative)
/// impl<S, I: IntoIterator<Item = S>> Tr<S> for I { }
/// impl<T> Tr<T> for Vec<T> { }
/// ```
///
/// The substs for the parent impl here are `[T, Vec<T>]`, which repeats `T`,
/// but `S` is constrained in the parent impl, so `parent_substs` is only
/// `[Vec<T>]`. This means we allow this impl.
fn check_duplicate_params<'tcx>(
    tcx: TyCtxt<'tcx>,
    impl1_substs: SubstsRef<'tcx>,
    parent_substs: &Vec<GenericArg<'tcx>>,
    span: Span,
) {
    let mut base_params = cgp::parameters_for(parent_substs, true);
    base_params.sort_by_key(|param| param.0);
    if let (_, [duplicate, ..]) = base_params.partition_dedup() {
        let param = impl1_substs[duplicate.0 as usize];
        tcx.sess
            .struct_span_err(span, &format!("specializing impl repeats parameter `{}`", param))
            .emit();
    }
}

/// Check that `'static` lifetimes are not introduced by the specializing impl.
///
/// For example forbid the following:
///
/// ```ignore (illustrative)
/// impl<A> Tr for A { }
/// impl Tr for &'static i32 { }
/// ```
fn check_static_lifetimes<'tcx>(
    tcx: TyCtxt<'tcx>,
    parent_substs: &Vec<GenericArg<'tcx>>,
    span: Span,
) {
    if tcx.any_free_region_meets(parent_substs, |r| r.is_static()) {
        tcx.sess.struct_span_err(span, "cannot specialize on `'static` lifetime").emit();
    }
}

/// Check whether predicates on the specializing impl (`impl1`) are allowed.
///
/// Each predicate `P` must be one of:
///
/// * Global (not reference any parameters).
/// * A `T: Tr` predicate where `Tr` is an always-applicable trait.
/// * Present on the base impl `impl2`.
///     * This check is done using the `trait_predicates_eq` function below.
/// * A well-formed predicate of a type argument of the trait being implemented,
///   including the `Self`-type.
#[instrument(level = "debug", skip(tcx))]
fn check_predicates<'tcx>(
    tcx: TyCtxt<'tcx>,
    impl1_def_id: LocalDefId,
    impl1_substs: SubstsRef<'tcx>,
    impl2_node: Node,
    impl2_substs: SubstsRef<'tcx>,
    span: Span,
) {
    let instantiated = tcx.predicates_of(impl1_def_id).instantiate(tcx, impl1_substs);
    let impl1_predicates: Vec<_> = traits::elaborate_predicates_with_span(
        tcx,
        std::iter::zip(
            instantiated.predicates,
            // Don't drop predicates (unsound!) because `spans` is too short
            instantiated.spans.into_iter().chain(std::iter::repeat(span)),
        ),
    )
    .map(|obligation| (obligation.predicate, obligation.cause.span))
    .collect();

    let mut impl2_predicates = if impl2_node.is_from_trait() {
        // Always applicable traits have to be always applicable without any
        // assumptions.
        Vec::new()
    } else {
        traits::elaborate_predicates(
            tcx,
            tcx.predicates_of(impl2_node.def_id())
                .instantiate(tcx, impl2_substs)
                .predicates
                .into_iter(),
        )
        .map(|obligation| obligation.predicate)
        .collect()
    };
    debug!(?impl1_predicates, ?impl2_predicates);

    // Since impls of always applicable traits don't get to assume anything, we
    // can also assume their supertraits apply.
    //
    // For example, we allow:
    //
    // #[rustc_specialization_trait]
    // trait AlwaysApplicable: Debug { }
    //
    // impl<T> Tr for T { }
    // impl<T: AlwaysApplicable> Tr for T { }
    //
    // Specializing on `AlwaysApplicable` allows also specializing on `Debug`
    // which is sound because we forbid impls like the following
    //
    // impl<D: Debug> AlwaysApplicable for D { }
    let always_applicable_traits = impl1_predicates.iter().copied().filter(|&(predicate, _)| {
        matches!(
            trait_predicate_kind(tcx, predicate),
            Some(TraitSpecializationKind::AlwaysApplicable)
        )
    });

    // Include the well-formed predicates of the type parameters of the impl.
    for arg in tcx.impl_trait_ref(impl1_def_id).unwrap().subst_identity().substs {
        let infcx = &tcx.infer_ctxt().build();
        let obligations =
            wf::obligations(infcx, tcx.param_env(impl1_def_id), impl1_def_id, 0, arg, span)
                .unwrap();

        assert!(!obligations.needs_infer());
        impl2_predicates.extend(
            traits::elaborate_obligations(tcx, obligations).map(|obligation| obligation.predicate),
        )
    }
    impl2_predicates.extend(
        traits::elaborate_predicates_with_span(tcx, always_applicable_traits)
            .map(|obligation| obligation.predicate),
    );

    for (predicate, span) in impl1_predicates {
        if !impl2_predicates.iter().any(|pred2| trait_predicates_eq(tcx, predicate, *pred2, span)) {
            check_specialization_on(tcx, predicate, span)
        }
    }
}

/// Checks if some predicate on the specializing impl (`predicate1`) is the same
/// as some predicate on the base impl (`predicate2`).
///
/// This basically just checks syntactic equivalence, but is a little more
/// forgiving since we want to equate `T: Tr` with `T: ~const Tr` so this can work:
///
/// ```ignore (illustrative)
/// #[rustc_specialization_trait]
/// trait Specialize { }
///
/// impl<T: Bound> Tr for T { }
/// impl<T: ~const Bound + Specialize> const Tr for T { }
/// ```
///
/// However, we *don't* want to allow the reverse, i.e., when the bound on the
/// specializing impl is not as const as the bound on the base impl:
///
/// ```ignore (illustrative)
/// impl<T: ~const Bound> const Tr for T { }
/// impl<T: Bound + Specialize> const Tr for T { } // should be T: ~const Bound
/// ```
///
/// So we make that check in this function and try to raise a helpful error message.
fn trait_predicates_eq<'tcx>(
    tcx: TyCtxt<'tcx>,
    predicate1: ty::Predicate<'tcx>,
    predicate2: ty::Predicate<'tcx>,
    span: Span,
) -> bool {
    let pred1_kind = predicate1.kind().skip_binder();
    let pred2_kind = predicate2.kind().skip_binder();
    let (trait_pred1, trait_pred2) = match (pred1_kind, pred2_kind) {
        (
            ty::PredicateKind::Clause(ty::Clause::Trait(pred1)),
            ty::PredicateKind::Clause(ty::Clause::Trait(pred2)),
        ) => (pred1, pred2),
        // Just use plain syntactic equivalence if either of the predicates aren't
        // trait predicates or have bound vars.
        _ => return predicate1 == predicate2,
    };

    let predicates_equal_modulo_constness = {
        let pred1_unconsted =
            ty::TraitPredicate { constness: ty::BoundConstness::NotConst, ..trait_pred1 };
        let pred2_unconsted =
            ty::TraitPredicate { constness: ty::BoundConstness::NotConst, ..trait_pred2 };
        pred1_unconsted == pred2_unconsted
    };

    if !predicates_equal_modulo_constness {
        return false;
    }

    // Check that the predicate on the specializing impl is at least as const as
    // the one on the base.
    match (trait_pred2.constness, trait_pred1.constness) {
        (ty::BoundConstness::ConstIfConst, ty::BoundConstness::NotConst) => {
            tcx.sess.struct_span_err(span, "missing `~const` qualifier for specialization").emit();
        }
        _ => {}
    }

    true
}

#[instrument(level = "debug", skip(tcx))]
fn check_specialization_on<'tcx>(tcx: TyCtxt<'tcx>, predicate: ty::Predicate<'tcx>, span: Span) {
    match predicate.kind().skip_binder() {
        // Global predicates are either always true or always false, so we
        // are fine to specialize on.
        _ if predicate.is_global() => (),
        // We allow specializing on explicitly marked traits with no associated
        // items.
        ty::PredicateKind::Clause(ty::Clause::Trait(ty::TraitPredicate {
            trait_ref,
            constness: _,
            polarity: _,
        })) => {
            if !matches!(
                trait_predicate_kind(tcx, predicate),
                Some(TraitSpecializationKind::Marker)
            ) {
                tcx.sess
                    .struct_span_err(
                        span,
                        &format!(
                            "cannot specialize on trait `{}`",
                            tcx.def_path_str(trait_ref.def_id),
                        ),
                    )
                    .emit();
            }
        }
        ty::PredicateKind::Clause(ty::Clause::Projection(ty::ProjectionPredicate {
            projection_ty,
            term,
        })) => {
            tcx.sess
                .struct_span_err(
                    span,
                    &format!("cannot specialize on associated type `{projection_ty} == {term}`",),
                )
                .emit();
        }
        ty::PredicateKind::Clause(ty::Clause::ConstArgHasType(..)) => {
            // FIXME(min_specialization), FIXME(const_generics):
            // It probably isn't right to allow _every_ `ConstArgHasType` but I am somewhat unsure
            // about the actual rules that would be sound. Can't just always error here because otherwise
            // std/core doesn't even compile as they have `const N: usize` in some specializing impls.
            //
            // While we do not support constructs like `<T, const N: T>` there is probably no risk of
            // soundness bugs, but when we support generic const parameter types this will need to be
            // revisited.
        }
        _ => {
            tcx.sess
                .struct_span_err(span, &format!("cannot specialize on predicate `{}`", predicate))
                .emit();
        }
    }
}

fn trait_predicate_kind<'tcx>(
    tcx: TyCtxt<'tcx>,
    predicate: ty::Predicate<'tcx>,
) -> Option<TraitSpecializationKind> {
    match predicate.kind().skip_binder() {
        ty::PredicateKind::Clause(ty::Clause::Trait(ty::TraitPredicate {
            trait_ref,
            constness: _,
            polarity: _,
        })) => Some(tcx.trait_def(trait_ref.def_id).specialization_kind),
        ty::PredicateKind::Clause(ty::Clause::RegionOutlives(_))
        | ty::PredicateKind::Clause(ty::Clause::TypeOutlives(_))
        | ty::PredicateKind::Clause(ty::Clause::Projection(_))
        | ty::PredicateKind::Clause(ty::Clause::ConstArgHasType(..))
        | ty::PredicateKind::AliasEq(..)
        | ty::PredicateKind::WellFormed(_)
        | ty::PredicateKind::Subtype(_)
        | ty::PredicateKind::Coerce(_)
        | ty::PredicateKind::ObjectSafe(_)
        | ty::PredicateKind::ClosureKind(..)
        | ty::PredicateKind::ConstEvaluatable(..)
        | ty::PredicateKind::ConstEquate(..)
        | ty::PredicateKind::Ambiguous
        | ty::PredicateKind::TypeWellFormedFromEnv(..) => None,
    }
}
