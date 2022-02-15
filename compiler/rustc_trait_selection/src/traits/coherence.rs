//! See Rustc Dev Guide chapters on [trait-resolution] and [trait-specialization] for more info on
//! how this works.
//!
//! [trait-resolution]: https://rustc-dev-guide.rust-lang.org/traits/resolution.html
//! [trait-specialization]: https://rustc-dev-guide.rust-lang.org/traits/specialization.html

use crate::infer::outlives::env::OutlivesEnvironment;
use crate::infer::{CombinedSnapshot, InferOk, RegionckMode};
use crate::traits::select::IntercrateAmbiguityCause;
use crate::traits::util::impl_trait_ref_and_oblig;
use crate::traits::SkipLeakCheck;
use crate::traits::{
    self, FulfillmentContext, Normalized, Obligation, ObligationCause, PredicateObligation,
    PredicateObligations, SelectionContext,
};
//use rustc_data_structures::fx::FxHashMap;
use rustc_hir::def_id::{DefId, LOCAL_CRATE};
use rustc_hir::CRATE_HIR_ID;
use rustc_infer::infer::TyCtxtInferExt;
use rustc_infer::traits::TraitEngine;
use rustc_middle::traits::specialization_graph::OverlapMode;
use rustc_middle::ty::fast_reject::{self, SimplifyParams};
use rustc_middle::ty::fold::TypeFoldable;
use rustc_middle::ty::subst::Subst;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_span::symbol::sym;
use rustc_span::DUMMY_SP;
use std::iter;

/// Whether we do the orphan check relative to this crate or
/// to some remote crate.
#[derive(Copy, Clone, Debug)]
enum InCrate {
    Local,
    Remote,
}

#[derive(Debug, Copy, Clone)]
pub enum Conflict {
    Upstream,
    Downstream,
}

pub struct OverlapResult<'tcx> {
    pub impl_header: ty::ImplHeader<'tcx>,
    pub intercrate_ambiguity_causes: Vec<IntercrateAmbiguityCause>,

    /// `true` if the overlap might've been permitted before the shift
    /// to universes.
    pub involves_placeholder: bool,
}

pub fn add_placeholder_note(err: &mut rustc_errors::DiagnosticBuilder<'_>) {
    err.note(
        "this behavior recently changed as a result of a bug fix; \
         see rust-lang/rust#56105 for details",
    );
}

/// If there are types that satisfy both impls, invokes `on_overlap`
/// with a suitably-freshened `ImplHeader` with those types
/// substituted. Otherwise, invokes `no_overlap`.
#[instrument(skip(tcx, skip_leak_check, on_overlap, no_overlap), level = "debug")]
pub fn overlapping_impls<F1, F2, R>(
    tcx: TyCtxt<'_>,
    impl1_def_id: DefId,
    impl2_def_id: DefId,
    skip_leak_check: SkipLeakCheck,
    overlap_mode: OverlapMode,
    on_overlap: F1,
    no_overlap: F2,
) -> R
where
    F1: FnOnce(OverlapResult<'_>) -> R,
    F2: FnOnce() -> R,
{
    // Before doing expensive operations like entering an inference context, do
    // a quick check via fast_reject to tell if the impl headers could possibly
    // unify.
    let impl1_ref = tcx.impl_trait_ref(impl1_def_id);
    let impl2_ref = tcx.impl_trait_ref(impl2_def_id);

    // Check if any of the input types definitely do not unify.
    if iter::zip(
        impl1_ref.iter().flat_map(|tref| tref.substs.types()),
        impl2_ref.iter().flat_map(|tref| tref.substs.types()),
    )
    .any(|(ty1, ty2)| {
        let t1 = fast_reject::simplify_type(tcx, ty1, SimplifyParams::No);
        let t2 = fast_reject::simplify_type(tcx, ty2, SimplifyParams::No);

        if let (Some(t1), Some(t2)) = (t1, t2) {
            // Simplified successfully
            t1 != t2
        } else {
            // Types might unify
            false
        }
    }) {
        // Some types involved are definitely different, so the impls couldn't possibly overlap.
        debug!("overlapping_impls: fast_reject early-exit");
        return no_overlap();
    }

    let overlaps = tcx.infer_ctxt().enter(|infcx| {
        let selcx = &mut SelectionContext::intercrate(&infcx);
        overlap(selcx, skip_leak_check, impl1_def_id, impl2_def_id, overlap_mode).is_some()
    });

    if !overlaps {
        return no_overlap();
    }

    // In the case where we detect an error, run the check again, but
    // this time tracking intercrate ambuiguity causes for better
    // diagnostics. (These take time and can lead to false errors.)
    tcx.infer_ctxt().enter(|infcx| {
        let selcx = &mut SelectionContext::intercrate(&infcx);
        selcx.enable_tracking_intercrate_ambiguity_causes();
        on_overlap(
            overlap(selcx, skip_leak_check, impl1_def_id, impl2_def_id, overlap_mode).unwrap(),
        )
    })
}

fn with_fresh_ty_vars<'cx, 'tcx>(
    selcx: &mut SelectionContext<'cx, 'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    impl_def_id: DefId,
) -> ty::ImplHeader<'tcx> {
    let tcx = selcx.tcx();
    let impl_substs = selcx.infcx().fresh_substs_for_item(DUMMY_SP, impl_def_id);

    let header = ty::ImplHeader {
        impl_def_id,
        self_ty: tcx.type_of(impl_def_id).subst(tcx, impl_substs),
        trait_ref: tcx.impl_trait_ref(impl_def_id).subst(tcx, impl_substs),
        predicates: tcx.predicates_of(impl_def_id).instantiate(tcx, impl_substs).predicates,
    };

    let Normalized { value: mut header, obligations } =
        traits::normalize(selcx, param_env, ObligationCause::dummy(), header);

    header.predicates.extend(obligations.into_iter().map(|o| o.predicate));
    header
}

/// Can both impl `a` and impl `b` be satisfied by a common type (including
/// where-clauses)? If so, returns an `ImplHeader` that unifies the two impls.
fn overlap<'cx, 'tcx>(
    selcx: &mut SelectionContext<'cx, 'tcx>,
    skip_leak_check: SkipLeakCheck,
    impl1_def_id: DefId,
    impl2_def_id: DefId,
    overlap_mode: OverlapMode,
) -> Option<OverlapResult<'tcx>> {
    debug!(
        "overlap(impl1_def_id={:?}, impl2_def_id={:?}, overlap_mode={:?})",
        impl1_def_id, impl2_def_id, overlap_mode
    );

    selcx.infcx().probe_maybe_skip_leak_check(skip_leak_check.is_yes(), |snapshot| {
        overlap_within_probe(
            selcx,
            skip_leak_check,
            impl1_def_id,
            impl2_def_id,
            overlap_mode,
            snapshot,
        )
    })
}

fn overlap_within_probe<'cx, 'tcx>(
    selcx: &mut SelectionContext<'cx, 'tcx>,
    skip_leak_check: SkipLeakCheck,
    impl1_def_id: DefId,
    impl2_def_id: DefId,
    overlap_mode: OverlapMode,
    snapshot: &CombinedSnapshot<'_, 'tcx>,
) -> Option<OverlapResult<'tcx>> {
    let infcx = selcx.infcx();

    if overlap_mode.use_negative_impl() {
        if negative_impl(selcx, impl1_def_id, impl2_def_id)
            || negative_impl(selcx, impl2_def_id, impl1_def_id)
        {
            return None;
        }
    }

    // For the purposes of this check, we don't bring any placeholder
    // types into scope; instead, we replace the generic types with
    // fresh type variables, and hence we do our evaluations in an
    // empty environment.
    let param_env = ty::ParamEnv::empty();

    let impl1_header = with_fresh_ty_vars(selcx, param_env, impl1_def_id);
    let impl2_header = with_fresh_ty_vars(selcx, param_env, impl2_def_id);

    let obligations = equate_impl_headers(selcx, &impl1_header, &impl2_header)?;
    debug!("overlap: unification check succeeded");

    if overlap_mode.use_implicit_negative() {
        if implicit_negative(selcx, param_env, &impl1_header, impl2_header, obligations) {
            return None;
        }
    }

    if !skip_leak_check.is_yes() {
        if infcx.leak_check(true, snapshot).is_err() {
            debug!("overlap: leak check failed");
            return None;
        }
    }

    let intercrate_ambiguity_causes = selcx.take_intercrate_ambiguity_causes();
    debug!("overlap: intercrate_ambiguity_causes={:#?}", intercrate_ambiguity_causes);

    let involves_placeholder =
        matches!(selcx.infcx().region_constraints_added_in_snapshot(snapshot), Some(true));

    let impl_header = selcx.infcx().resolve_vars_if_possible(impl1_header);
    Some(OverlapResult { impl_header, intercrate_ambiguity_causes, involves_placeholder })
}

fn equate_impl_headers<'cx, 'tcx>(
    selcx: &mut SelectionContext<'cx, 'tcx>,
    impl1_header: &ty::ImplHeader<'tcx>,
    impl2_header: &ty::ImplHeader<'tcx>,
) -> Option<PredicateObligations<'tcx>> {
    // Do `a` and `b` unify? If not, no overlap.
    debug!("equate_impl_headers(impl1_header={:?}, impl2_header={:?}", impl1_header, impl2_header);
    selcx
        .infcx()
        .at(&ObligationCause::dummy(), ty::ParamEnv::empty())
        .eq_impl_headers(impl1_header, impl2_header)
        .map(|infer_ok| infer_ok.obligations)
        .ok()
}

/// Given impl1 and impl2 check if both impls can be satisfied by a common type (including
/// where-clauses) If so, return false, otherwise return true, they are disjoint.
fn implicit_negative<'cx, 'tcx>(
    selcx: &mut SelectionContext<'cx, 'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    impl1_header: &ty::ImplHeader<'tcx>,
    impl2_header: ty::ImplHeader<'tcx>,
    obligations: PredicateObligations<'tcx>,
) -> bool {
    // There's no overlap if obligations are unsatisfiable or if the obligation negated is
    // satisfied.
    //
    // For example, given these two impl headers:
    //
    // `impl<'a> From<&'a str> for Box<dyn Error>`
    // `impl<E> From<E> for Box<dyn Error> where E: Error`
    //
    // So we have:
    //
    // `Box<dyn Error>: From<&'?a str>`
    // `Box<dyn Error>: From<?E>`
    //
    // After equating the two headers:
    //
    // `Box<dyn Error> = Box<dyn Error>`
    // So, `?E = &'?a str` and then given the where clause `&'?a str: Error`.
    //
    // If the obligation `&'?a str: Error` holds, it means that there's overlap. If that doesn't
    // hold we need to check if `&'?a str: !Error` holds, if doesn't hold there's overlap because
    // at some point an impl for `&'?a str: Error` could be added.
    debug!(
        "implicit_negative(impl1_header={:?}, impl2_header={:?}, obligations={:?})",
        impl1_header, impl2_header, obligations
    );
    let infcx = selcx.infcx();
    let opt_failing_obligation = impl1_header
        .predicates
        .iter()
        .copied()
        .chain(impl2_header.predicates)
        .map(|p| infcx.resolve_vars_if_possible(p))
        .map(|p| Obligation {
            cause: ObligationCause::dummy(),
            param_env,
            recursion_depth: 0,
            predicate: p,
        })
        .chain(obligations)
        .find(|o| !selcx.predicate_may_hold_fatal(o));

    if let Some(failing_obligation) = opt_failing_obligation {
        debug!("overlap: obligation unsatisfiable {:?}", failing_obligation);
        true
    } else {
        false
    }
}

/// Given impl1 and impl2 check if both impls are never satisfied by a common type (including
/// where-clauses) If so, return true, they are disjoint and false otherwise.
fn negative_impl<'cx, 'tcx>(
    selcx: &mut SelectionContext<'cx, 'tcx>,
    impl1_def_id: DefId,
    impl2_def_id: DefId,
) -> bool {
    debug!("negative_impl(impl1_def_id={:?}, impl2_def_id={:?})", impl1_def_id, impl2_def_id);
    let tcx = selcx.infcx().tcx;

    // create a parameter environment corresponding to a (placeholder) instantiation of impl1
    let impl1_env = tcx.param_env(impl1_def_id);
    let impl1_trait_ref = tcx.impl_trait_ref(impl1_def_id).unwrap();

    // Create an infcx, taking the predicates of impl1 as assumptions:
    tcx.infer_ctxt().enter(|infcx| {
        // Normalize the trait reference. The WF rules ought to ensure
        // that this always succeeds.
        let impl1_trait_ref = match traits::fully_normalize(
            &infcx,
            FulfillmentContext::new(),
            ObligationCause::dummy(),
            impl1_env,
            impl1_trait_ref,
        ) {
            Ok(impl1_trait_ref) => impl1_trait_ref,
            Err(err) => {
                bug!("failed to fully normalize {:?}: {:?}", impl1_trait_ref, err);
            }
        };

        // Attempt to prove that impl2 applies, given all of the above.
        let selcx = &mut SelectionContext::new(&infcx);
        let impl2_substs = infcx.fresh_substs_for_item(DUMMY_SP, impl2_def_id);
        let (impl2_trait_ref, obligations) =
            impl_trait_ref_and_oblig(selcx, impl1_env, impl2_def_id, impl2_substs);

        // do the impls unify? If not, not disjoint.
        let more_obligations = match infcx
            .at(&ObligationCause::dummy(), impl1_env)
            .eq(impl1_trait_ref, impl2_trait_ref)
        {
            Ok(InferOk { obligations, .. }) => obligations,
            Err(_) => {
                debug!(
                    "explicit_disjoint: {:?} does not unify with {:?}",
                    impl1_trait_ref, impl2_trait_ref
                );
                return false;
            }
        };

        let opt_failing_obligation = obligations
            .into_iter()
            .chain(more_obligations)
            .find(|o| negative_impl_exists(selcx, impl1_env, impl1_def_id, o));

        if let Some(failing_obligation) = opt_failing_obligation {
            debug!("overlap: obligation unsatisfiable {:?}", failing_obligation);
            true
        } else {
            false
        }
    })
}

fn negative_impl_exists<'cx, 'tcx>(
    selcx: &SelectionContext<'cx, 'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    region_context: DefId,
    o: &PredicateObligation<'tcx>,
) -> bool {
    let infcx = &selcx.infcx().fork();
    let tcx = infcx.tcx;
    o.flip_polarity(tcx)
        .map(|o| {
            let mut fulfillment_cx = FulfillmentContext::new();
            fulfillment_cx.register_predicate_obligation(infcx, o);

            let errors = fulfillment_cx.select_all_or_error(infcx);
            if !errors.is_empty() {
                return false;
            }

            let mut outlives_env = OutlivesEnvironment::new(param_env);
            // FIXME -- add "assumed to be well formed" types into the `outlives_env`

            // "Save" the accumulated implied bounds into the outlives environment
            // (due to the FIXME above, there aren't any, but this step is still needed).
            // The "body id" is given as `CRATE_HIR_ID`, which is the same body-id used
            // by the "dummy" causes elsewhere (body-id is only relevant when checking
            // function bodies with closures).
            outlives_env.save_implied_bounds(CRATE_HIR_ID);

            infcx.process_registered_region_obligations(
                outlives_env.region_bound_pairs_map(),
                Some(tcx.lifetimes.re_root_empty),
                param_env,
            );

            let errors =
                infcx.resolve_regions(region_context, &outlives_env, RegionckMode::default());
            if !errors.is_empty() {
                return false;
            }

            true
        })
        .unwrap_or(false)
}

pub fn trait_ref_is_knowable<'tcx>(
    tcx: TyCtxt<'tcx>,
    trait_ref: ty::TraitRef<'tcx>,
) -> Option<Conflict> {
    debug!("trait_ref_is_knowable(trait_ref={:?})", trait_ref);
    if orphan_check_trait_ref(tcx, trait_ref, InCrate::Remote).is_ok() {
        // A downstream or cousin crate is allowed to implement some
        // substitution of this trait-ref.
        return Some(Conflict::Downstream);
    }

    if trait_ref_is_local_or_fundamental(tcx, trait_ref) {
        // This is a local or fundamental trait, so future-compatibility
        // is no concern. We know that downstream/cousin crates are not
        // allowed to implement a substitution of this trait ref, which
        // means impls could only come from dependencies of this crate,
        // which we already know about.
        return None;
    }

    // This is a remote non-fundamental trait, so if another crate
    // can be the "final owner" of a substitution of this trait-ref,
    // they are allowed to implement it future-compatibly.
    //
    // However, if we are a final owner, then nobody else can be,
    // and if we are an intermediate owner, then we don't care
    // about future-compatibility, which means that we're OK if
    // we are an owner.
    if orphan_check_trait_ref(tcx, trait_ref, InCrate::Local).is_ok() {
        debug!("trait_ref_is_knowable: orphan check passed");
        None
    } else {
        debug!("trait_ref_is_knowable: nonlocal, nonfundamental, unowned");
        Some(Conflict::Upstream)
    }
}

pub fn trait_ref_is_local_or_fundamental<'tcx>(
    tcx: TyCtxt<'tcx>,
    trait_ref: ty::TraitRef<'tcx>,
) -> bool {
    trait_ref.def_id.krate == LOCAL_CRATE || tcx.has_attr(trait_ref.def_id, sym::fundamental)
}

pub enum OrphanCheckErr<'tcx> {
    NonLocalInputType(Vec<(Ty<'tcx>, bool /* Is this the first input type? */)>),
    UncoveredTy(Ty<'tcx>, Option<Ty<'tcx>>),
}

/// Checks the coherence orphan rules. `impl_def_id` should be the
/// `DefId` of a trait impl. To pass, either the trait must be local, or else
/// two conditions must be satisfied:
///
/// 1. All type parameters in `Self` must be "covered" by some local type constructor.
/// 2. Some local type must appear in `Self`.
pub fn orphan_check(tcx: TyCtxt<'_>, impl_def_id: DefId) -> Result<(), OrphanCheckErr<'_>> {
    debug!("orphan_check({:?})", impl_def_id);

    // We only except this routine to be invoked on implementations
    // of a trait, not inherent implementations.
    let trait_ref = tcx.impl_trait_ref(impl_def_id).unwrap();
    debug!("orphan_check: trait_ref={:?}", trait_ref);

    // If the *trait* is local to the crate, ok.
    if trait_ref.def_id.is_local() {
        debug!("trait {:?} is local to current crate", trait_ref.def_id);
        return Ok(());
    }

    orphan_check_trait_ref(tcx, trait_ref, InCrate::Local)
}

/// Checks whether a trait-ref is potentially implementable by a crate.
///
/// The current rule is that a trait-ref orphan checks in a crate C:
///
/// 1. Order the parameters in the trait-ref in subst order - Self first,
///    others linearly (e.g., `<U as Foo<V, W>>` is U < V < W).
/// 2. Of these type parameters, there is at least one type parameter
///    in which, walking the type as a tree, you can reach a type local
///    to C where all types in-between are fundamental types. Call the
///    first such parameter the "local key parameter".
///     - e.g., `Box<LocalType>` is OK, because you can visit LocalType
///       going through `Box`, which is fundamental.
///     - similarly, `FundamentalPair<Vec<()>, Box<LocalType>>` is OK for
///       the same reason.
///     - but (knowing that `Vec<T>` is non-fundamental, and assuming it's
///       not local), `Vec<LocalType>` is bad, because `Vec<->` is between
///       the local type and the type parameter.
/// 3. Before this local type, no generic type parameter of the impl must
///    be reachable through fundamental types.
///     - e.g. `impl<T> Trait<LocalType> for Vec<T>` is fine, as `Vec` is not fundamental.
///     - while `impl<T> Trait<LocalType for Box<T>` results in an error, as `T` is
///       reachable through the fundamental type `Box`.
/// 4. Every type in the local key parameter not known in C, going
///    through the parameter's type tree, must appear only as a subtree of
///    a type local to C, with only fundamental types between the type
///    local to C and the local key parameter.
///     - e.g., `Vec<LocalType<T>>>` (or equivalently `Box<Vec<LocalType<T>>>`)
///     is bad, because the only local type with `T` as a subtree is
///     `LocalType<T>`, and `Vec<->` is between it and the type parameter.
///     - similarly, `FundamentalPair<LocalType<T>, T>` is bad, because
///     the second occurrence of `T` is not a subtree of *any* local type.
///     - however, `LocalType<Vec<T>>` is OK, because `T` is a subtree of
///     `LocalType<Vec<T>>`, which is local and has no types between it and
///     the type parameter.
///
/// The orphan rules actually serve several different purposes:
///
/// 1. They enable link-safety - i.e., 2 mutually-unknowing crates (where
///    every type local to one crate is unknown in the other) can't implement
///    the same trait-ref. This follows because it can be seen that no such
///    type can orphan-check in 2 such crates.
///
///    To check that a local impl follows the orphan rules, we check it in
///    InCrate::Local mode, using type parameters for the "generic" types.
///
/// 2. They ground negative reasoning for coherence. If a user wants to
///    write both a conditional blanket impl and a specific impl, we need to
///    make sure they do not overlap. For example, if we write
///    ```
///    impl<T> IntoIterator for Vec<T>
///    impl<T: Iterator> IntoIterator for T
///    ```
///    We need to be able to prove that `Vec<$0>: !Iterator` for every type $0.
///    We can observe that this holds in the current crate, but we need to make
///    sure this will also hold in all unknown crates (both "independent" crates,
///    which we need for link-safety, and also child crates, because we don't want
///    child crates to get error for impl conflicts in a *dependency*).
///
///    For that, we only allow negative reasoning if, for every assignment to the
///    inference variables, every unknown crate would get an orphan error if they
///    try to implement this trait-ref. To check for this, we use InCrate::Remote
///    mode. That is sound because we already know all the impls from known crates.
///
/// 3. For non-`#[fundamental]` traits, they guarantee that parent crates can
///    add "non-blanket" impls without breaking negative reasoning in dependent
///    crates. This is the "rebalancing coherence" (RFC 1023) restriction.
///
///    For that, we only a allow crate to perform negative reasoning on
///    non-local-non-`#[fundamental]` only if there's a local key parameter as per (2).
///
///    Because we never perform negative reasoning generically (coherence does
///    not involve type parameters), this can be interpreted as doing the full
///    orphan check (using InCrate::Local mode), substituting non-local known
///    types for all inference variables.
///
///    This allows for crates to future-compatibly add impls as long as they
///    can't apply to types with a key parameter in a child crate - applying
///    the rules, this basically means that every type parameter in the impl
///    must appear behind a non-fundamental type (because this is not a
///    type-system requirement, crate owners might also go for "semantic
///    future-compatibility" involving things such as sealed traits, but
///    the above requirement is sufficient, and is necessary in "open world"
///    cases).
///
/// Note that this function is never called for types that have both type
/// parameters and inference variables.
fn orphan_check_trait_ref<'tcx>(
    tcx: TyCtxt<'tcx>,
    trait_ref: ty::TraitRef<'tcx>,
    in_crate: InCrate,
) -> Result<(), OrphanCheckErr<'tcx>> {
    debug!("orphan_check_trait_ref(trait_ref={:?}, in_crate={:?})", trait_ref, in_crate);

    if trait_ref.needs_infer() && trait_ref.needs_subst() {
        bug!(
            "can't orphan check a trait ref with both params and inference variables {:?}",
            trait_ref
        );
    }

    // Given impl<P1..=Pn> Trait<T1..=Tn> for T0, an impl is valid only
    // if at least one of the following is true:
    //
    // - Trait is a local trait
    // (already checked in orphan_check prior to calling this function)
    // - All of
    //     - At least one of the types T0..=Tn must be a local type.
    //      Let Ti be the first such type.
    //     - No uncovered type parameters P1..=Pn may appear in T0..Ti (excluding Ti)
    //
    fn uncover_fundamental_ty<'tcx>(
        tcx: TyCtxt<'tcx>,
        ty: Ty<'tcx>,
        in_crate: InCrate,
    ) -> Vec<Ty<'tcx>> {
        // FIXME: this is currently somewhat overly complicated,
        // but fixing this requires a more complicated refactor.
        if !contained_non_local_types(tcx, ty, in_crate).is_empty() {
            if let Some(inner_tys) = fundamental_ty_inner_tys(tcx, ty) {
                return inner_tys
                    .flat_map(|ty| uncover_fundamental_ty(tcx, ty, in_crate))
                    .collect();
            }
        }

        vec![ty]
    }

    let mut non_local_spans = vec![];
    for (i, input_ty) in trait_ref
        .substs
        .types()
        .flat_map(|ty| uncover_fundamental_ty(tcx, ty, in_crate))
        .enumerate()
    {
        debug!("orphan_check_trait_ref: check ty `{:?}`", input_ty);
        let non_local_tys = contained_non_local_types(tcx, input_ty, in_crate);
        if non_local_tys.is_empty() {
            debug!("orphan_check_trait_ref: ty_is_local `{:?}`", input_ty);
            return Ok(());
        } else if let ty::Param(_) = input_ty.kind() {
            debug!("orphan_check_trait_ref: uncovered ty: `{:?}`", input_ty);
            let local_type = trait_ref
                .substs
                .types()
                .flat_map(|ty| uncover_fundamental_ty(tcx, ty, in_crate))
                .find(|ty| ty_is_local_constructor(*ty, in_crate));

            debug!("orphan_check_trait_ref: uncovered ty local_type: `{:?}`", local_type);

            return Err(OrphanCheckErr::UncoveredTy(input_ty, local_type));
        }

        non_local_spans.extend(non_local_tys.into_iter().map(|input_ty| (input_ty, i == 0)));
    }
    // If we exit above loop, never found a local type.
    debug!("orphan_check_trait_ref: no local type");
    Err(OrphanCheckErr::NonLocalInputType(non_local_spans))
}

/// Returns a list of relevant non-local types for `ty`.
///
/// This is just `ty` itself unless `ty` is `#[fundamental]`,
/// in which case we recursively look into this type.
///
/// If `ty` is local itself, this method returns an empty `Vec`.
///
/// # Examples
///
/// - `u32` is not local, so this returns `[u32]`.
/// - for `Foo<u32>`, where `Foo` is a local type, this returns `[]`.
/// - `&mut u32` returns `[u32]`, as `&mut` is a fundamental type, similar to `Box`.
/// - `Box<Foo<u32>>` returns `[]`, as `Box` is a fundamental type and `Foo` is local.
fn contained_non_local_types<'tcx>(
    tcx: TyCtxt<'tcx>,
    ty: Ty<'tcx>,
    in_crate: InCrate,
) -> Vec<Ty<'tcx>> {
    if ty_is_local_constructor(ty, in_crate) {
        Vec::new()
    } else {
        match fundamental_ty_inner_tys(tcx, ty) {
            Some(inner_tys) => {
                inner_tys.flat_map(|ty| contained_non_local_types(tcx, ty, in_crate)).collect()
            }
            None => vec![ty],
        }
    }
}

/// For `#[fundamental]` ADTs and `&T` / `&mut T`, returns `Some` with the
/// type parameters of the ADT, or `T`, respectively. For non-fundamental
/// types, returns `None`.
fn fundamental_ty_inner_tys<'tcx>(
    tcx: TyCtxt<'tcx>,
    ty: Ty<'tcx>,
) -> Option<impl Iterator<Item = Ty<'tcx>>> {
    let (first_ty, rest_tys) = match *ty.kind() {
        ty::Ref(_, ty, _) => (ty, ty::subst::InternalSubsts::empty().types()),
        ty::Adt(def, substs) if def.is_fundamental() => {
            let mut types = substs.types();

            // FIXME(eddyb) actually validate `#[fundamental]` up-front.
            match types.next() {
                None => {
                    tcx.sess.span_err(
                        tcx.def_span(def.did),
                        "`#[fundamental]` requires at least one type parameter",
                    );

                    return None;
                }

                Some(first_ty) => (first_ty, types),
            }
        }
        _ => return None,
    };

    Some(iter::once(first_ty).chain(rest_tys))
}

fn def_id_is_local(def_id: DefId, in_crate: InCrate) -> bool {
    match in_crate {
        // The type is local to *this* crate - it will not be
        // local in any other crate.
        InCrate::Remote => false,
        InCrate::Local => def_id.is_local(),
    }
}

fn ty_is_local_constructor(ty: Ty<'_>, in_crate: InCrate) -> bool {
    debug!("ty_is_local_constructor({:?})", ty);

    match *ty.kind() {
        ty::Bool
        | ty::Char
        | ty::Int(..)
        | ty::Uint(..)
        | ty::Float(..)
        | ty::Str
        | ty::FnDef(..)
        | ty::FnPtr(_)
        | ty::Array(..)
        | ty::Slice(..)
        | ty::RawPtr(..)
        | ty::Ref(..)
        | ty::Never
        | ty::Tuple(..)
        | ty::Param(..)
        | ty::Projection(..) => false,

        ty::Placeholder(..) | ty::Bound(..) | ty::Infer(..) => match in_crate {
            InCrate::Local => false,
            // The inference variable might be unified with a local
            // type in that remote crate.
            InCrate::Remote => true,
        },

        ty::Adt(def, _) => def_id_is_local(def.did, in_crate),
        ty::Foreign(did) => def_id_is_local(did, in_crate),
        ty::Opaque(..) => {
            // This merits some explanation.
            // Normally, opaque types are not involed when performing
            // coherence checking, since it is illegal to directly
            // implement a trait on an opaque type. However, we might
            // end up looking at an opaque type during coherence checking
            // if an opaque type gets used within another type (e.g. as
            // a type parameter). This requires us to decide whether or
            // not an opaque type should be considered 'local' or not.
            //
            // We choose to treat all opaque types as non-local, even
            // those that appear within the same crate. This seems
            // somewhat surprising at first, but makes sense when
            // you consider that opaque types are supposed to hide
            // the underlying type *within the same crate*. When an
            // opaque type is used from outside the module
            // where it is declared, it should be impossible to observe
            // anything about it other than the traits that it implements.
            //
            // The alternative would be to look at the underlying type
            // to determine whether or not the opaque type itself should
            // be considered local. However, this could make it a breaking change
            // to switch the underlying ('defining') type from a local type
            // to a remote type. This would violate the rule that opaque
            // types should be completely opaque apart from the traits
            // that they implement, so we don't use this behavior.
            false
        }

        ty::Closure(..) => {
            // Similar to the `Opaque` case (#83613).
            false
        }

        ty::Dynamic(ref tt, ..) => {
            if let Some(principal) = tt.principal() {
                def_id_is_local(principal.def_id(), in_crate)
            } else {
                false
            }
        }

        ty::Error(_) => true,

        ty::Generator(..) | ty::GeneratorWitness(..) => {
            bug!("ty_is_local invoked on unexpected type: {:?}", ty)
        }
    }
}
