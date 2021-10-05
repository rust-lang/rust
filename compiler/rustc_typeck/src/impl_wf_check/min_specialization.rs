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
//! ```rust
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

use rustc_data_structures::fx::FxHashSet;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_infer::infer::outlives::env::OutlivesEnvironment;
use rustc_infer::infer::{InferCtxt, RegionckMode, TyCtxtInferExt};
use rustc_infer::traits::specialization_graph::Node;
use rustc_middle::ty::subst::{GenericArg, InternalSubsts, SubstsRef};
use rustc_middle::ty::trait_def::TraitSpecializationKind;
use rustc_middle::ty::{self, TyCtxt, TypeFoldable};
use rustc_span::Span;
use rustc_trait_selection::traits::{self, translate_substs, wf};

pub(super) fn check_min_specialization(tcx: TyCtxt<'_>, impl_def_id: DefId, span: Span) {
    if let Some(node) = parent_specialization_node(tcx, impl_def_id) {
        tcx.infer_ctxt().enter(|infcx| {
            check_always_applicable(&infcx, impl_def_id, node, span);
        });
    }
}

fn parent_specialization_node(tcx: TyCtxt<'_>, impl1_def_id: DefId) -> Option<Node> {
    let trait_ref = tcx.impl_trait_ref(impl1_def_id)?;
    let trait_def = tcx.trait_def(trait_ref.def_id);

    let impl2_node = trait_def.ancestors(tcx, impl1_def_id).ok()?.nth(1)?;

    let always_applicable_trait =
        matches!(trait_def.specialization_kind, TraitSpecializationKind::AlwaysApplicable);
    if impl2_node.is_from_trait() && !always_applicable_trait {
        // Implementing a normal trait isn't a specialization.
        return None;
    }
    Some(impl2_node)
}

/// Check that `impl1` is a sound specialization
fn check_always_applicable(
    infcx: &InferCtxt<'_, '_>,
    impl1_def_id: DefId,
    impl2_node: Node,
    span: Span,
) {
    if let Some((impl1_substs, impl2_substs)) =
        get_impl_substs(infcx, impl1_def_id, impl2_node, span)
    {
        let impl2_def_id = impl2_node.def_id();
        debug!(
            "check_always_applicable(\nimpl1_def_id={:?},\nimpl2_def_id={:?},\nimpl2_substs={:?}\n)",
            impl1_def_id, impl2_def_id, impl2_substs
        );

        let tcx = infcx.tcx;

        let parent_substs = if impl2_node.is_from_trait() {
            impl2_substs.to_vec()
        } else {
            unconstrained_parent_impl_substs(tcx, impl2_def_id, impl2_substs)
        };

        check_static_lifetimes(tcx, &parent_substs, span);
        check_duplicate_params(tcx, impl1_substs, &parent_substs, span);

        check_predicates(
            infcx,
            impl1_def_id.expect_local(),
            impl1_substs,
            impl2_node,
            impl2_substs,
            span,
        );
    }
}

/// Given a specializing impl `impl1`, and the base impl `impl2`, returns two
/// substitutions `(S1, S2)` that equate their trait references. The returned
/// types are expressed in terms of the generics of `impl1`.
///
/// Example
///
/// impl<A, B> Foo<A> for B { /* impl2 */ }
/// impl<C> Foo<Vec<C>> for C { /* impl1 */ }
///
/// Would return `S1 = [C]` and `S2 = [Vec<C>, C]`.
fn get_impl_substs<'tcx>(
    infcx: &InferCtxt<'_, 'tcx>,
    impl1_def_id: DefId,
    impl2_node: Node,
    span: Span,
) -> Option<(SubstsRef<'tcx>, SubstsRef<'tcx>)> {
    let tcx = infcx.tcx;
    let param_env = tcx.param_env(impl1_def_id);

    let impl1_substs = InternalSubsts::identity_for_item(tcx, impl1_def_id);
    let impl2_substs = translate_substs(infcx, param_env, impl1_def_id, impl1_substs, impl2_node);

    // Conservatively use an empty `ParamEnv`.
    let outlives_env = OutlivesEnvironment::new(ty::ParamEnv::empty());
    infcx.resolve_regions_and_report_errors(impl1_def_id, &outlives_env, RegionckMode::default());
    let impl2_substs = match infcx.fully_resolve(impl2_substs) {
        Ok(s) => s,
        Err(_) => {
            tcx.sess.struct_span_err(span, "could not resolve substs on overridden impl").emit();
            return None;
        }
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
    let impl_trait_ref = tcx.impl_trait_ref(impl_def_id);

    // Unfortunately the functions in `constrained_generic_parameters` don't do
    // what we want here. We want only a list of constrained parameters while
    // the functions in `cgp` add the constrained parameters to a list of
    // unconstrained parameters.
    for (predicate, _) in impl_generic_predicates.predicates.iter() {
        if let ty::PredicateKind::Projection(proj) = predicate.kind().skip_binder() {
            let projection_ty = proj.projection_ty;
            let projected_ty = proj.ty;

            let unbound_trait_ref = projection_ty.trait_ref(tcx);
            if Some(unbound_trait_ref) == impl_trait_ref {
                continue;
            }

            unconstrained_parameters.extend(cgp::parameters_for(tcx, &projection_ty, true));

            for param in cgp::parameters_for(tcx, &projected_ty, false) {
                if !unconstrained_parameters.contains(&param) {
                    constrained_params.insert(param.0);
                }
            }

            unconstrained_parameters.extend(cgp::parameters_for(tcx, &projected_ty, true));
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
/// impl<A> Tr for A { }
/// impl<B> Tr for (B, B) { }
///
/// Note that only consider the unconstrained parameters of the base impl:
///
/// impl<S, I: IntoIterator<Item = S>> Tr<S> for I { }
/// impl<T> Tr<T> for Vec<T> { }
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
    let mut base_params = cgp::parameters_for(tcx, parent_substs, true);
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
/// impl<A> Tr for A { }
/// impl Tr for &'static i32 { }
fn check_static_lifetimes<'tcx>(
    tcx: TyCtxt<'tcx>,
    parent_substs: &Vec<GenericArg<'tcx>>,
    span: Span,
) {
    if tcx.any_free_region_meets(parent_substs, |r| *r == ty::ReStatic) {
        tcx.sess.struct_span_err(span, "cannot specialize on `'static` lifetime").emit();
    }
}

/// Check whether predicates on the specializing impl (`impl1`) are allowed.
///
/// Each predicate `P` must be:
///
/// * global (not reference any parameters)
/// * `T: Tr` predicate where `Tr` is an always-applicable trait
/// * on the base `impl impl2`
///     * Currently this check is done using syntactic equality, which is
///       conservative but generally sufficient.
/// * a well-formed predicate of a type argument of the trait being implemented,
///   including the `Self`-type.
fn check_predicates<'tcx>(
    infcx: &InferCtxt<'_, 'tcx>,
    impl1_def_id: LocalDefId,
    impl1_substs: SubstsRef<'tcx>,
    impl2_node: Node,
    impl2_substs: SubstsRef<'tcx>,
    span: Span,
) {
    let tcx = infcx.tcx;
    let impl1_predicates: Vec<_> = traits::elaborate_predicates(
        tcx,
        tcx.predicates_of(impl1_def_id).instantiate(tcx, impl1_substs).predicates.into_iter(),
    )
    .map(|obligation| obligation.predicate)
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
    debug!(
        "check_always_applicable(\nimpl1_predicates={:?},\nimpl2_predicates={:?}\n)",
        impl1_predicates, impl2_predicates,
    );

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
    let always_applicable_traits = impl1_predicates.iter().copied().filter(|&predicate| {
        matches!(
            trait_predicate_kind(tcx, predicate),
            Some(TraitSpecializationKind::AlwaysApplicable)
        )
    });

    // Include the well-formed predicates of the type parameters of the impl.
    for arg in tcx.impl_trait_ref(impl1_def_id).unwrap().substs {
        if let Some(obligations) = wf::obligations(
            infcx,
            tcx.param_env(impl1_def_id),
            tcx.hir().local_def_id_to_hir_id(impl1_def_id),
            0,
            arg,
            span,
        ) {
            impl2_predicates.extend(
                traits::elaborate_obligations(tcx, obligations)
                    .map(|obligation| obligation.predicate),
            )
        }
    }
    impl2_predicates.extend(
        traits::elaborate_predicates(tcx, always_applicable_traits)
            .map(|obligation| obligation.predicate),
    );

    for predicate in impl1_predicates {
        if !impl2_predicates.contains(&predicate) {
            check_specialization_on(tcx, predicate, span)
        }
    }
}

fn check_specialization_on<'tcx>(tcx: TyCtxt<'tcx>, predicate: ty::Predicate<'tcx>, span: Span) {
    debug!("can_specialize_on(predicate = {:?})", predicate);
    match predicate.kind().skip_binder() {
        // Global predicates are either always true or always false, so we
        // are fine to specialize on.
        _ if predicate.is_global(tcx) => (),
        // We allow specializing on explicitly marked traits with no associated
        // items.
        ty::PredicateKind::Trait(ty::TraitPredicate {
            trait_ref,
            constness: ty::BoundConstness::NotConst,
        }) => {
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
                    .emit()
            }
        }
        _ => tcx
            .sess
            .struct_span_err(span, &format!("cannot specialize on `{:?}`", predicate))
            .emit(),
    }
}

fn trait_predicate_kind<'tcx>(
    tcx: TyCtxt<'tcx>,
    predicate: ty::Predicate<'tcx>,
) -> Option<TraitSpecializationKind> {
    match predicate.kind().skip_binder() {
        ty::PredicateKind::Trait(ty::TraitPredicate {
            trait_ref,
            constness: ty::BoundConstness::NotConst,
        }) => Some(tcx.trait_def(trait_ref.def_id).specialization_kind),
        ty::PredicateKind::Trait(_)
        | ty::PredicateKind::RegionOutlives(_)
        | ty::PredicateKind::TypeOutlives(_)
        | ty::PredicateKind::Projection(_)
        | ty::PredicateKind::WellFormed(_)
        | ty::PredicateKind::Subtype(_)
        | ty::PredicateKind::Coerce(_)
        | ty::PredicateKind::ObjectSafe(_)
        | ty::PredicateKind::ClosureKind(..)
        | ty::PredicateKind::ConstEvaluatable(..)
        | ty::PredicateKind::ConstEquate(..)
        | ty::PredicateKind::TypeWellFormedFromEnv(..) => None,
    }
}
