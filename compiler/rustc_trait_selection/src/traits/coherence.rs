//! See Rustc Dev Guide chapters on [trait-resolution] and [trait-specialization] for more info on
//! how this works.
//!
//! [trait-resolution]: https://rustc-dev-guide.rust-lang.org/traits/resolution.html
//! [trait-specialization]: https://rustc-dev-guide.rust-lang.org/traits/specialization.html

use crate::infer::outlives::env::OutlivesEnvironment;
use crate::infer::InferOk;
use crate::regions::InferCtxtRegionExt;
use crate::solve::inspect::{InspectGoal, ProofTreeInferCtxtExt, ProofTreeVisitor};
use crate::solve::{deeply_normalize_for_diagnostics, inspect, FulfillmentCtxt};
use crate::traits::engine::TraitEngineExt as _;
use crate::traits::select::IntercrateAmbiguityCause;
use crate::traits::structural_normalize::StructurallyNormalizeExt;
use crate::traits::NormalizeExt;
use crate::traits::SkipLeakCheck;
use crate::traits::{
    Obligation, ObligationCause, PredicateObligation, PredicateObligations, SelectionContext,
};
use rustc_data_structures::fx::FxIndexSet;
use rustc_errors::{DiagnosticBuilder, EmissionGuarantee};
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{DefId, LOCAL_CRATE};
use rustc_infer::infer::{DefineOpaqueTypes, InferCtxt, TyCtxtInferExt};
use rustc_infer::traits::{util, TraitEngine, TraitEngineExt};
use rustc_middle::traits::query::NoSolution;
use rustc_middle::traits::solve::{CandidateSource, Certainty, Goal};
use rustc_middle::traits::specialization_graph::OverlapMode;
use rustc_middle::traits::DefiningAnchor;
use rustc_middle::ty::fast_reject::{DeepRejectCtxt, TreatParams};
use rustc_middle::ty::visit::{TypeVisitable, TypeVisitableExt};
use rustc_middle::ty::{self, Ty, TyCtxt, TypeSuperVisitable, TypeVisitor};
use rustc_span::symbol::sym;
use rustc_span::DUMMY_SP;
use std::fmt::Debug;
use std::ops::ControlFlow;

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
    pub intercrate_ambiguity_causes: FxIndexSet<IntercrateAmbiguityCause<'tcx>>,

    /// `true` if the overlap might've been permitted before the shift
    /// to universes.
    pub involves_placeholder: bool,
}

pub fn add_placeholder_note<G: EmissionGuarantee>(err: &mut DiagnosticBuilder<'_, G>) {
    err.note(
        "this behavior recently changed as a result of a bug fix; \
         see rust-lang/rust#56105 for details",
    );
}

#[derive(Debug, Clone, Copy)]
enum TrackAmbiguityCauses {
    Yes,
    No,
}

impl TrackAmbiguityCauses {
    fn is_yes(self) -> bool {
        match self {
            TrackAmbiguityCauses::Yes => true,
            TrackAmbiguityCauses::No => false,
        }
    }
}

/// If there are types that satisfy both impls, returns `Some`
/// with a suitably-freshened `ImplHeader` with those types
/// instantiated. Otherwise, returns `None`.
#[instrument(skip(tcx, skip_leak_check), level = "debug")]
pub fn overlapping_impls(
    tcx: TyCtxt<'_>,
    impl1_def_id: DefId,
    impl2_def_id: DefId,
    skip_leak_check: SkipLeakCheck,
    overlap_mode: OverlapMode,
) -> Option<OverlapResult<'_>> {
    // Before doing expensive operations like entering an inference context, do
    // a quick check via fast_reject to tell if the impl headers could possibly
    // unify.
    let drcx = DeepRejectCtxt { treat_obligation_params: TreatParams::AsCandidateKey };
    let impl1_ref = tcx.impl_trait_ref(impl1_def_id);
    let impl2_ref = tcx.impl_trait_ref(impl2_def_id);
    let may_overlap = match (impl1_ref, impl2_ref) {
        (Some(a), Some(b)) => drcx.args_may_unify(a.skip_binder().args, b.skip_binder().args),
        (None, None) => {
            let self_ty1 = tcx.type_of(impl1_def_id).skip_binder();
            let self_ty2 = tcx.type_of(impl2_def_id).skip_binder();
            drcx.types_may_unify(self_ty1, self_ty2)
        }
        _ => bug!("unexpected impls: {impl1_def_id:?} {impl2_def_id:?}"),
    };

    if !may_overlap {
        // Some types involved are definitely different, so the impls couldn't possibly overlap.
        debug!("overlapping_impls: fast_reject early-exit");
        return None;
    }

    let _overlap_with_bad_diagnostics = overlap(
        tcx,
        TrackAmbiguityCauses::No,
        skip_leak_check,
        impl1_def_id,
        impl2_def_id,
        overlap_mode,
    )?;

    // In the case where we detect an error, run the check again, but
    // this time tracking intercrate ambiguity causes for better
    // diagnostics. (These take time and can lead to false errors.)
    let overlap = overlap(
        tcx,
        TrackAmbiguityCauses::Yes,
        skip_leak_check,
        impl1_def_id,
        impl2_def_id,
        overlap_mode,
    )
    .unwrap();
    Some(overlap)
}

fn fresh_impl_header<'tcx>(infcx: &InferCtxt<'tcx>, impl_def_id: DefId) -> ty::ImplHeader<'tcx> {
    let tcx = infcx.tcx;
    let impl_args = infcx.fresh_args_for_item(DUMMY_SP, impl_def_id);

    ty::ImplHeader {
        impl_def_id,
        impl_args,
        self_ty: tcx.type_of(impl_def_id).instantiate(tcx, impl_args),
        trait_ref: tcx.impl_trait_ref(impl_def_id).map(|i| i.instantiate(tcx, impl_args)),
        predicates: tcx
            .predicates_of(impl_def_id)
            .instantiate(tcx, impl_args)
            .iter()
            .map(|(c, _)| c.as_predicate())
            .collect(),
    }
}

fn fresh_impl_header_normalized<'tcx>(
    infcx: &InferCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    impl_def_id: DefId,
) -> ty::ImplHeader<'tcx> {
    let header = fresh_impl_header(infcx, impl_def_id);

    let InferOk { value: mut header, obligations } =
        infcx.at(&ObligationCause::dummy(), param_env).normalize(header);

    header.predicates.extend(obligations.into_iter().map(|o| o.predicate));
    header
}

/// Can both impl `a` and impl `b` be satisfied by a common type (including
/// where-clauses)? If so, returns an `ImplHeader` that unifies the two impls.
#[instrument(level = "debug", skip(tcx))]
fn overlap<'tcx>(
    tcx: TyCtxt<'tcx>,
    track_ambiguity_causes: TrackAmbiguityCauses,
    skip_leak_check: SkipLeakCheck,
    impl1_def_id: DefId,
    impl2_def_id: DefId,
    overlap_mode: OverlapMode,
) -> Option<OverlapResult<'tcx>> {
    if overlap_mode.use_negative_impl() {
        if impl_intersection_has_negative_obligation(tcx, impl1_def_id, impl2_def_id)
            || impl_intersection_has_negative_obligation(tcx, impl2_def_id, impl1_def_id)
        {
            return None;
        }
    }

    let infcx = tcx
        .infer_ctxt()
        .with_opaque_type_inference(DefiningAnchor::Bubble)
        .skip_leak_check(skip_leak_check.is_yes())
        .intercrate(true)
        .with_next_trait_solver(tcx.next_trait_solver_in_coherence())
        .build();
    let selcx = &mut SelectionContext::with_treat_inductive_cycle_as_ambig(&infcx);
    if track_ambiguity_causes.is_yes() {
        selcx.enable_tracking_intercrate_ambiguity_causes();
    }

    // For the purposes of this check, we don't bring any placeholder
    // types into scope; instead, we replace the generic types with
    // fresh type variables, and hence we do our evaluations in an
    // empty environment.
    let param_env = ty::ParamEnv::empty();

    let impl1_header = fresh_impl_header_normalized(selcx.infcx, param_env, impl1_def_id);
    let impl2_header = fresh_impl_header_normalized(selcx.infcx, param_env, impl2_def_id);

    // Equate the headers to find their intersection (the general type, with infer vars,
    // that may apply both impls).
    let mut obligations =
        equate_impl_headers(selcx.infcx, param_env, &impl1_header, &impl2_header)?;
    debug!("overlap: unification check succeeded");

    obligations.extend(
        [&impl1_header.predicates, &impl2_header.predicates].into_iter().flatten().map(
            |&predicate| Obligation::new(infcx.tcx, ObligationCause::dummy(), param_env, predicate),
        ),
    );

    if overlap_mode.use_implicit_negative() {
        if let Some(_failing_obligation) =
            impl_intersection_has_impossible_obligation(selcx, &obligations)
        {
            return None;
        }
    }

    // We toggle the `leak_check` by using `skip_leak_check` when constructing the
    // inference context, so this may be a noop.
    if infcx.leak_check(ty::UniverseIndex::ROOT, None).is_err() {
        debug!("overlap: leak check failed");
        return None;
    }

    let intercrate_ambiguity_causes = if !overlap_mode.use_implicit_negative() {
        Default::default()
    } else if infcx.next_trait_solver() {
        compute_intercrate_ambiguity_causes(&infcx, &obligations)
    } else {
        selcx.take_intercrate_ambiguity_causes()
    };

    debug!("overlap: intercrate_ambiguity_causes={:#?}", intercrate_ambiguity_causes);
    let involves_placeholder = infcx
        .inner
        .borrow_mut()
        .unwrap_region_constraints()
        .data()
        .constraints
        .iter()
        .any(|c| c.0.involves_placeholders());

    let mut impl_header = infcx.resolve_vars_if_possible(impl1_header);

    // Deeply normalize the impl header for diagnostics, ignoring any errors if this fails.
    if infcx.next_trait_solver() {
        impl_header = deeply_normalize_for_diagnostics(&infcx, param_env, impl_header);
    }

    Some(OverlapResult { impl_header, intercrate_ambiguity_causes, involves_placeholder })
}

#[instrument(level = "debug", skip(infcx), ret)]
fn equate_impl_headers<'tcx>(
    infcx: &InferCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    impl1: &ty::ImplHeader<'tcx>,
    impl2: &ty::ImplHeader<'tcx>,
) -> Option<PredicateObligations<'tcx>> {
    let result =
        match (impl1.trait_ref, impl2.trait_ref) {
            (Some(impl1_ref), Some(impl2_ref)) => infcx
                .at(&ObligationCause::dummy(), param_env)
                .eq(DefineOpaqueTypes::Yes, impl1_ref, impl2_ref),
            (None, None) => infcx.at(&ObligationCause::dummy(), param_env).eq(
                DefineOpaqueTypes::Yes,
                impl1.self_ty,
                impl2.self_ty,
            ),
            _ => bug!("equate_impl_headers given mismatched impl kinds"),
        };

    result.map(|infer_ok| infer_ok.obligations).ok()
}

/// Check if both impls can be satisfied by a common type by considering whether
/// any of either impl's obligations is not known to hold.
///
/// For example, given these two impls:
///     `impl From<MyLocalType> for Box<dyn Error>` (in my crate)
///     `impl<E> From<E> for Box<dyn Error> where E: Error` (in libstd)
///
/// After replacing both impl headers with inference vars (which happens before
/// this function is called), we get:
///     `Box<dyn Error>: From<MyLocalType>`
///     `Box<dyn Error>: From<?E>`
///
/// This gives us `?E = MyLocalType`. We then certainly know that `MyLocalType: Error`
/// never holds in intercrate mode since a local impl does not exist, and a
/// downstream impl cannot be added -- therefore can consider the intersection
/// of the two impls above to be empty.
///
/// Importantly, this works even if there isn't a `impl !Error for MyLocalType`.
fn impl_intersection_has_impossible_obligation<'a, 'cx, 'tcx>(
    selcx: &mut SelectionContext<'cx, 'tcx>,
    obligations: &'a [PredicateObligation<'tcx>],
) -> Option<PredicateObligation<'tcx>> {
    let infcx = selcx.infcx;

    if infcx.next_trait_solver() {
        let mut fulfill_cx = FulfillmentCtxt::new(infcx);
        fulfill_cx.register_predicate_obligations(infcx, obligations.iter().cloned());

        // We only care about the obligations that are *definitely* true errors.
        // Ambiguities do not prove the disjointness of two impls.
        let mut errors = fulfill_cx.select_where_possible(infcx);
        errors.pop().map(|err| err.obligation)
    } else {
        obligations.iter().cloned().find(|obligation| {
            // We use `evaluate_root_obligation` to correctly track intercrate
            // ambiguity clauses. We cannot use this in the new solver.
            let evaluation_result = selcx.evaluate_root_obligation(obligation);

            match evaluation_result {
                Ok(result) => !result.may_apply(),
                // If overflow occurs, we need to conservatively treat the goal as possibly holding,
                // since there can be instantiations of this goal that don't overflow and result in
                // success. This isn't much of a problem in the old solver, since we treat overflow
                // fatally (this still can be encountered: <https://github.com/rust-lang/rust/issues/105231>),
                // but in the new solver, this is very important for correctness, since overflow
                // *must* be treated as ambiguity for completeness.
                Err(_overflow) => false,
            }
        })
    }
}

/// Check if both impls can be satisfied by a common type by considering whether
/// any of first impl's obligations is known not to hold *via a negative predicate*.
///
/// For example, given these two impls:
///     `struct MyCustomBox<T: ?Sized>(Box<T>);`
///     `impl From<&str> for MyCustomBox<dyn Error>` (in my crate)
///     `impl<E> From<E> for MyCustomBox<dyn Error> where E: Error` (in my crate)
///
/// After replacing the second impl's header with inference vars, we get:
///     `MyCustomBox<dyn Error>: From<&str>`
///     `MyCustomBox<dyn Error>: From<?E>`
///
/// This gives us `?E = &str`. We then try to prove the first impl's predicates
/// after negating, giving us `&str: !Error`. This is a negative impl provided by
/// libstd, and therefore we can guarantee for certain that libstd will never add
/// a positive impl for `&str: Error` (without it being a breaking change).
fn impl_intersection_has_negative_obligation(
    tcx: TyCtxt<'_>,
    impl1_def_id: DefId,
    impl2_def_id: DefId,
) -> bool {
    debug!("negative_impl(impl1_def_id={:?}, impl2_def_id={:?})", impl1_def_id, impl2_def_id);

    // N.B. We need to unify impl headers *with* intercrate mode, even if proving negative predicates
    // do not need intercrate mode enabled.
    let ref infcx = tcx.infer_ctxt().intercrate(true).with_next_trait_solver(true).build();
    let root_universe = infcx.universe();
    assert_eq!(root_universe, ty::UniverseIndex::ROOT);

    let impl1_header = fresh_impl_header(infcx, impl1_def_id);
    let param_env =
        ty::EarlyBinder::bind(tcx.param_env(impl1_def_id)).instantiate(tcx, impl1_header.impl_args);

    let impl2_header = fresh_impl_header(infcx, impl2_def_id);

    // Equate the headers to find their intersection (the general type, with infer vars,
    // that may apply both impls).
    let Some(equate_obligations) =
        equate_impl_headers(infcx, param_env, &impl1_header, &impl2_header)
    else {
        return false;
    };

    // FIXME(with_negative_coherence): the infcx has constraints from equating
    // the impl headers. We should use these constraints as assumptions, not as
    // requirements, when proving the negated where clauses below.
    drop(equate_obligations);
    drop(infcx.take_registered_region_obligations());
    drop(infcx.take_and_reset_region_constraints());

    plug_infer_with_placeholders(
        infcx,
        root_universe,
        (impl1_header.impl_args, impl2_header.impl_args),
    );
    let param_env = infcx.resolve_vars_if_possible(param_env);

    util::elaborate(tcx, tcx.predicates_of(impl2_def_id).instantiate(tcx, impl2_header.impl_args))
        .any(|(clause, _)| try_prove_negated_where_clause(infcx, clause, param_env))
}

fn plug_infer_with_placeholders<'tcx>(
    infcx: &InferCtxt<'tcx>,
    universe: ty::UniverseIndex,
    value: impl TypeVisitable<TyCtxt<'tcx>>,
) {
    struct PlugInferWithPlaceholder<'a, 'tcx> {
        infcx: &'a InferCtxt<'tcx>,
        universe: ty::UniverseIndex,
        var: ty::BoundVar,
    }

    impl<'tcx> PlugInferWithPlaceholder<'_, 'tcx> {
        fn next_var(&mut self) -> ty::BoundVar {
            let var = self.var;
            self.var = self.var + 1;
            var
        }
    }

    impl<'tcx> TypeVisitor<TyCtxt<'tcx>> for PlugInferWithPlaceholder<'_, 'tcx> {
        fn visit_ty(&mut self, ty: Ty<'tcx>) -> ControlFlow<Self::BreakTy> {
            let ty = self.infcx.shallow_resolve(ty);
            if ty.is_ty_var() {
                let Ok(InferOk { value: (), obligations }) =
                    self.infcx.at(&ObligationCause::dummy(), ty::ParamEnv::empty()).eq(
                        DefineOpaqueTypes::No,
                        ty,
                        Ty::new_placeholder(
                            self.infcx.tcx,
                            ty::Placeholder {
                                universe: self.universe,
                                bound: ty::BoundTy {
                                    var: self.next_var(),
                                    kind: ty::BoundTyKind::Anon,
                                },
                            },
                        ),
                    )
                else {
                    bug!("we always expect to be able to plug an infer var with placeholder")
                };
                assert_eq!(obligations, &[]);
                ControlFlow::Continue(())
            } else {
                ty.super_visit_with(self)
            }
        }

        fn visit_const(&mut self, ct: ty::Const<'tcx>) -> ControlFlow<Self::BreakTy> {
            let ct = self.infcx.shallow_resolve(ct);
            if ct.is_ct_infer() {
                let Ok(InferOk { value: (), obligations }) =
                    self.infcx.at(&ObligationCause::dummy(), ty::ParamEnv::empty()).eq(
                        DefineOpaqueTypes::No,
                        ct,
                        ty::Const::new_placeholder(
                            self.infcx.tcx,
                            ty::Placeholder { universe: self.universe, bound: self.next_var() },
                            ct.ty(),
                        ),
                    )
                else {
                    bug!("we always expect to be able to plug an infer var with placeholder")
                };
                assert_eq!(obligations, &[]);
                ControlFlow::Continue(())
            } else {
                ct.super_visit_with(self)
            }
        }

        fn visit_region(&mut self, r: ty::Region<'tcx>) -> ControlFlow<Self::BreakTy> {
            if let ty::ReVar(vid) = *r {
                let r = self
                    .infcx
                    .inner
                    .borrow_mut()
                    .unwrap_region_constraints()
                    .opportunistic_resolve_var(self.infcx.tcx, vid);
                if r.is_var() {
                    let Ok(InferOk { value: (), obligations }) =
                        self.infcx.at(&ObligationCause::dummy(), ty::ParamEnv::empty()).eq(
                            DefineOpaqueTypes::No,
                            r,
                            ty::Region::new_placeholder(
                                self.infcx.tcx,
                                ty::Placeholder {
                                    universe: self.universe,
                                    bound: ty::BoundRegion {
                                        var: self.next_var(),
                                        kind: ty::BoundRegionKind::BrAnon,
                                    },
                                },
                            ),
                        )
                    else {
                        bug!("we always expect to be able to plug an infer var with placeholder")
                    };
                    assert_eq!(obligations, &[]);
                }
            }
            ControlFlow::Continue(())
        }
    }

    value.visit_with(&mut PlugInferWithPlaceholder {
        infcx,
        universe,
        var: ty::BoundVar::from_u32(0),
    });
}

fn try_prove_negated_where_clause<'tcx>(
    root_infcx: &InferCtxt<'tcx>,
    clause: ty::Clause<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
) -> bool {
    let Some(negative_predicate) = clause.as_predicate().flip_polarity(root_infcx.tcx) else {
        return false;
    };

    // N.B. We don't need to use intercrate mode here because we're trying to prove
    // the *existence* of a negative goal, not the non-existence of a positive goal.
    // Without this, we over-eagerly register coherence ambiguity candidates when
    // impl candidates do exist.
    let ref infcx = root_infcx.fork_with_intercrate(false);
    let mut fulfill_cx = FulfillmentCtxt::new(infcx);

    fulfill_cx.register_predicate_obligation(
        infcx,
        Obligation::new(infcx.tcx, ObligationCause::dummy(), param_env, negative_predicate),
    );
    if !fulfill_cx.select_all_or_error(infcx).is_empty() {
        return false;
    }

    // FIXME: We could use the assumed_wf_types from both impls, I think,
    // if that wasn't implemented just for LocalDefId, and we'd need to do
    // the normalization ourselves since this is totally fallible...
    let outlives_env = OutlivesEnvironment::new(param_env);

    let errors = infcx.resolve_regions(&outlives_env);
    if !errors.is_empty() {
        return false;
    }

    true
}

/// Returns whether all impls which would apply to the `trait_ref`
/// e.g. `Ty: Trait<Arg>` are already known in the local crate.
///
/// This both checks whether any downstream or sibling crates could
/// implement it and whether an upstream crate can add this impl
/// without breaking backwards compatibility.
#[instrument(level = "debug", skip(tcx, lazily_normalize_ty), ret)]
pub fn trait_ref_is_knowable<'tcx, E: Debug>(
    tcx: TyCtxt<'tcx>,
    trait_ref: ty::TraitRef<'tcx>,
    mut lazily_normalize_ty: impl FnMut(Ty<'tcx>) -> Result<Ty<'tcx>, E>,
) -> Result<Result<(), Conflict>, E> {
    if orphan_check_trait_ref(trait_ref, InCrate::Remote, &mut lazily_normalize_ty)?.is_ok() {
        // A downstream or cousin crate is allowed to implement some
        // generic parameters of this trait-ref.
        return Ok(Err(Conflict::Downstream));
    }

    if trait_ref_is_local_or_fundamental(tcx, trait_ref) {
        // This is a local or fundamental trait, so future-compatibility
        // is no concern. We know that downstream/cousin crates are not
        // allowed to implement a generic parameter of this trait ref,
        // which means impls could only come from dependencies of this
        // crate, which we already know about.
        return Ok(Ok(()));
    }

    // This is a remote non-fundamental trait, so if another crate
    // can be the "final owner" of the generic parameters of this trait-ref,
    // they are allowed to implement it future-compatibly.
    //
    // However, if we are a final owner, then nobody else can be,
    // and if we are an intermediate owner, then we don't care
    // about future-compatibility, which means that we're OK if
    // we are an owner.
    if orphan_check_trait_ref(trait_ref, InCrate::Local, &mut lazily_normalize_ty)?.is_ok() {
        Ok(Ok(()))
    } else {
        Ok(Err(Conflict::Upstream))
    }
}

pub fn trait_ref_is_local_or_fundamental<'tcx>(
    tcx: TyCtxt<'tcx>,
    trait_ref: ty::TraitRef<'tcx>,
) -> bool {
    trait_ref.def_id.krate == LOCAL_CRATE || tcx.has_attr(trait_ref.def_id, sym::fundamental)
}

#[derive(Debug)]
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
#[instrument(level = "debug", skip(tcx), ret)]
pub fn orphan_check(tcx: TyCtxt<'_>, impl_def_id: DefId) -> Result<(), OrphanCheckErr<'_>> {
    // We only except this routine to be invoked on implementations
    // of a trait, not inherent implementations.
    let trait_ref = tcx.impl_trait_ref(impl_def_id).unwrap().instantiate_identity();
    debug!(?trait_ref);

    // If the *trait* is local to the crate, ok.
    if trait_ref.def_id.is_local() {
        debug!("trait {:?} is local to current crate", trait_ref.def_id);
        return Ok(());
    }

    orphan_check_trait_ref::<!>(trait_ref, InCrate::Local, |ty| Ok(ty)).unwrap()
}

/// Checks whether a trait-ref is potentially implementable by a crate.
///
/// The current rule is that a trait-ref orphan checks in a crate C:
///
/// 1. Order the parameters in the trait-ref in generic parameters order
/// - Self first, others linearly (e.g., `<U as Foo<V, W>>` is U < V < W).
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
///     - while `impl<T> Trait<LocalType> for Box<T>` results in an error, as `T` is
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
///    ```ignore (illustrative)
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
///    orphan check (using InCrate::Local mode), instantiating non-local known
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
#[instrument(level = "trace", skip(lazily_normalize_ty), ret)]
fn orphan_check_trait_ref<'tcx, E: Debug>(
    trait_ref: ty::TraitRef<'tcx>,
    in_crate: InCrate,
    lazily_normalize_ty: impl FnMut(Ty<'tcx>) -> Result<Ty<'tcx>, E>,
) -> Result<Result<(), OrphanCheckErr<'tcx>>, E> {
    if trait_ref.has_infer() && trait_ref.has_param() {
        bug!(
            "can't orphan check a trait ref with both params and inference variables {:?}",
            trait_ref
        );
    }

    let mut checker = OrphanChecker::new(in_crate, lazily_normalize_ty);
    Ok(match trait_ref.visit_with(&mut checker) {
        ControlFlow::Continue(()) => Err(OrphanCheckErr::NonLocalInputType(checker.non_local_tys)),
        ControlFlow::Break(OrphanCheckEarlyExit::NormalizationFailure(err)) => return Err(err),
        ControlFlow::Break(OrphanCheckEarlyExit::ParamTy(ty)) => {
            // Does there exist some local type after the `ParamTy`.
            checker.search_first_local_ty = true;
            if let Some(OrphanCheckEarlyExit::LocalTy(local_ty)) =
                trait_ref.visit_with(&mut checker).break_value()
            {
                Err(OrphanCheckErr::UncoveredTy(ty, Some(local_ty)))
            } else {
                Err(OrphanCheckErr::UncoveredTy(ty, None))
            }
        }
        ControlFlow::Break(OrphanCheckEarlyExit::LocalTy(_)) => Ok(()),
    })
}

struct OrphanChecker<'tcx, F> {
    in_crate: InCrate,
    in_self_ty: bool,
    lazily_normalize_ty: F,
    /// Ignore orphan check failures and exclusively search for the first
    /// local type.
    search_first_local_ty: bool,
    non_local_tys: Vec<(Ty<'tcx>, bool)>,
}

impl<'tcx, F, E> OrphanChecker<'tcx, F>
where
    F: FnOnce(Ty<'tcx>) -> Result<Ty<'tcx>, E>,
{
    fn new(in_crate: InCrate, lazily_normalize_ty: F) -> Self {
        OrphanChecker {
            in_crate,
            in_self_ty: true,
            lazily_normalize_ty,
            search_first_local_ty: false,
            non_local_tys: Vec::new(),
        }
    }

    fn found_non_local_ty(&mut self, t: Ty<'tcx>) -> ControlFlow<OrphanCheckEarlyExit<'tcx, E>> {
        self.non_local_tys.push((t, self.in_self_ty));
        ControlFlow::Continue(())
    }

    fn found_param_ty(&mut self, t: Ty<'tcx>) -> ControlFlow<OrphanCheckEarlyExit<'tcx, E>> {
        if self.search_first_local_ty {
            ControlFlow::Continue(())
        } else {
            ControlFlow::Break(OrphanCheckEarlyExit::ParamTy(t))
        }
    }

    fn def_id_is_local(&mut self, def_id: DefId) -> bool {
        match self.in_crate {
            InCrate::Local => def_id.is_local(),
            InCrate::Remote => false,
        }
    }
}

enum OrphanCheckEarlyExit<'tcx, E> {
    NormalizationFailure(E),
    ParamTy(Ty<'tcx>),
    LocalTy(Ty<'tcx>),
}

impl<'tcx, F, E> TypeVisitor<TyCtxt<'tcx>> for OrphanChecker<'tcx, F>
where
    F: FnMut(Ty<'tcx>) -> Result<Ty<'tcx>, E>,
{
    type BreakTy = OrphanCheckEarlyExit<'tcx, E>;
    fn visit_region(&mut self, _r: ty::Region<'tcx>) -> ControlFlow<Self::BreakTy> {
        ControlFlow::Continue(())
    }

    fn visit_ty(&mut self, ty: Ty<'tcx>) -> ControlFlow<Self::BreakTy> {
        // Need to lazily normalize here in with `-Znext-solver=coherence`.
        let ty = match (self.lazily_normalize_ty)(ty) {
            Ok(ty) => ty,
            Err(err) => return ControlFlow::Break(OrphanCheckEarlyExit::NormalizationFailure(err)),
        };

        let result = match *ty.kind() {
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
            | ty::Never
            | ty::Tuple(..)
            | ty::Alias(ty::Projection | ty::Inherent | ty::Weak, ..) => {
                self.found_non_local_ty(ty)
            }

            ty::Param(..) => self.found_param_ty(ty),

            ty::Placeholder(..) | ty::Bound(..) | ty::Infer(..) => match self.in_crate {
                InCrate::Local => self.found_non_local_ty(ty),
                // The inference variable might be unified with a local
                // type in that remote crate.
                InCrate::Remote => ControlFlow::Break(OrphanCheckEarlyExit::LocalTy(ty)),
            },

            // For fundamental types, we just look inside of them.
            ty::Ref(_, ty, _) => ty.visit_with(self),
            ty::Adt(def, args) => {
                if self.def_id_is_local(def.did()) {
                    ControlFlow::Break(OrphanCheckEarlyExit::LocalTy(ty))
                } else if def.is_fundamental() {
                    args.visit_with(self)
                } else {
                    self.found_non_local_ty(ty)
                }
            }
            ty::Foreign(def_id) => {
                if self.def_id_is_local(def_id) {
                    ControlFlow::Break(OrphanCheckEarlyExit::LocalTy(ty))
                } else {
                    self.found_non_local_ty(ty)
                }
            }
            ty::Dynamic(tt, ..) => {
                let principal = tt.principal().map(|p| p.def_id());
                if principal.is_some_and(|p| self.def_id_is_local(p)) {
                    ControlFlow::Break(OrphanCheckEarlyExit::LocalTy(ty))
                } else {
                    self.found_non_local_ty(ty)
                }
            }
            ty::Error(_) => ControlFlow::Break(OrphanCheckEarlyExit::LocalTy(ty)),
            ty::Closure(did, ..) | ty::CoroutineClosure(did, ..) | ty::Coroutine(did, ..) => {
                if self.def_id_is_local(did) {
                    ControlFlow::Break(OrphanCheckEarlyExit::LocalTy(ty))
                } else {
                    self.found_non_local_ty(ty)
                }
            }
            // This should only be created when checking whether we have to check whether some
            // auto trait impl applies. There will never be multiple impls, so we can just
            // act as if it were a local type here.
            ty::CoroutineWitness(..) => ControlFlow::Break(OrphanCheckEarlyExit::LocalTy(ty)),
            ty::Alias(ty::Opaque, ..) => {
                // This merits some explanation.
                // Normally, opaque types are not involved when performing
                // coherence checking, since it is illegal to directly
                // implement a trait on an opaque type. However, we might
                // end up looking at an opaque type during coherence checking
                // if an opaque type gets used within another type (e.g. as
                // the type of a field) when checking for auto trait or `Sized`
                // impls. This requires us to decide whether or not an opaque
                // type should be considered 'local' or not.
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
                self.found_non_local_ty(ty)
            }
        };
        // A bit of a hack, the `OrphanChecker` is only used to visit a `TraitRef`, so
        // the first type we visit is always the self type.
        self.in_self_ty = false;
        result
    }

    /// All possible values for a constant parameter already exist
    /// in the crate defining the trait, so they are always non-local[^1].
    ///
    /// Because there's no way to have an impl where the first local
    /// generic argument is a constant, we also don't have to fail
    /// the orphan check when encountering a parameter or a generic constant.
    ///
    /// This means that we can completely ignore constants during the orphan check.
    ///
    /// See `tests/ui/coherence/const-generics-orphan-check-ok.rs` for examples.
    ///
    /// [^1]: This might not hold for function pointers or trait objects in the future.
    /// As these should be quite rare as const arguments and especially rare as impl
    /// parameters, allowing uncovered const parameters in impls seems more useful
    /// than allowing `impl<T> Trait<local_fn_ptr, T> for i32` to compile.
    fn visit_const(&mut self, _c: ty::Const<'tcx>) -> ControlFlow<Self::BreakTy> {
        ControlFlow::Continue(())
    }
}

/// Compute the `intercrate_ambiguity_causes` for the new solver using
/// "proof trees".
///
/// This is a bit scuffed but seems to be good enough, at least
/// when looking at UI tests. Given that it is only used to improve
/// diagnostics this is good enough. We can always improve it once there
/// are test cases where it is currently not enough.
fn compute_intercrate_ambiguity_causes<'tcx>(
    infcx: &InferCtxt<'tcx>,
    obligations: &[PredicateObligation<'tcx>],
) -> FxIndexSet<IntercrateAmbiguityCause<'tcx>> {
    let mut causes: FxIndexSet<IntercrateAmbiguityCause<'tcx>> = Default::default();

    for obligation in obligations {
        search_ambiguity_causes(infcx, obligation.clone().into(), &mut causes);
    }

    causes
}

struct AmbiguityCausesVisitor<'a, 'tcx> {
    causes: &'a mut FxIndexSet<IntercrateAmbiguityCause<'tcx>>,
}

impl<'a, 'tcx> ProofTreeVisitor<'tcx> for AmbiguityCausesVisitor<'a, 'tcx> {
    type BreakTy = !;
    fn visit_goal(&mut self, goal: &InspectGoal<'_, 'tcx>) -> ControlFlow<Self::BreakTy> {
        let infcx = goal.infcx();
        for cand in goal.candidates() {
            cand.visit_nested(self)?;
        }
        // When searching for intercrate ambiguity causes, we only need to look
        // at ambiguous goals, as for others the coherence unknowable candidate
        // was irrelevant.
        match goal.result() {
            Ok(Certainty::Maybe(_)) => {}
            Ok(Certainty::Yes) | Err(NoSolution) => return ControlFlow::Continue(()),
        }

        let Goal { param_env, predicate } = goal.goal();

        // For bound predicates we simply call `infcx.enter_forall`
        // and then prove the resulting predicate as a nested goal.
        let trait_ref = match predicate.kind().no_bound_vars() {
            Some(ty::PredicateKind::Clause(ty::ClauseKind::Trait(tr))) => tr.trait_ref,
            Some(ty::PredicateKind::Clause(ty::ClauseKind::Projection(proj)))
                if matches!(
                    infcx.tcx.def_kind(proj.projection_ty.def_id),
                    DefKind::AssocTy | DefKind::AssocConst
                ) =>
            {
                proj.projection_ty.trait_ref(infcx.tcx)
            }
            _ => return ControlFlow::Continue(()),
        };

        // Add ambiguity causes for reservation impls.
        for cand in goal.candidates() {
            if let inspect::ProbeKind::TraitCandidate {
                source: CandidateSource::Impl(def_id),
                result: Ok(_),
            } = cand.kind()
            {
                if let ty::ImplPolarity::Reservation = infcx.tcx.impl_polarity(def_id) {
                    let message = infcx
                        .tcx
                        .get_attr(def_id, sym::rustc_reservation_impl)
                        .and_then(|a| a.value_str());
                    if let Some(message) = message {
                        self.causes.insert(IntercrateAmbiguityCause::ReservationImpl { message });
                    }
                }
            }
        }

        // Add ambiguity causes for unknowable goals.
        let mut ambiguity_cause = None;
        for cand in goal.candidates() {
            // FIXME: boiiii, using string comparisions here sure is scuffed.
            if let inspect::ProbeKind::MiscCandidate {
                name: "coherence unknowable",
                result: Ok(_),
            } = cand.kind()
            {
                let lazily_normalize_ty = |ty: Ty<'tcx>| {
                    let mut fulfill_cx = <dyn TraitEngine<'tcx>>::new(infcx);
                    if matches!(ty.kind(), ty::Alias(..)) {
                        // FIXME(-Znext-solver=coherence): we currently don't
                        // normalize opaque types here, resulting in diverging behavior
                        // for TAITs.
                        match infcx
                            .at(&ObligationCause::dummy(), param_env)
                            .structurally_normalize(ty, &mut *fulfill_cx)
                        {
                            Ok(ty) => Ok(ty),
                            Err(_errs) => Err(()),
                        }
                    } else {
                        Ok(ty)
                    }
                };

                infcx.probe(|_| {
                    match trait_ref_is_knowable(infcx.tcx, trait_ref, lazily_normalize_ty) {
                        Err(()) => {}
                        Ok(Ok(())) => warn!("expected an unknowable trait ref: {trait_ref:?}"),
                        Ok(Err(conflict)) => {
                            if !trait_ref.references_error() {
                                // Normalize the trait ref for diagnostics, ignoring any errors if this fails.
                                let trait_ref =
                                    deeply_normalize_for_diagnostics(infcx, param_env, trait_ref);

                                let self_ty = trait_ref.self_ty();
                                let self_ty = self_ty.has_concrete_skeleton().then(|| self_ty);
                                ambiguity_cause = Some(match conflict {
                                    Conflict::Upstream => {
                                        IntercrateAmbiguityCause::UpstreamCrateUpdate {
                                            trait_ref,
                                            self_ty,
                                        }
                                    }
                                    Conflict::Downstream => {
                                        IntercrateAmbiguityCause::DownstreamCrate {
                                            trait_ref,
                                            self_ty,
                                        }
                                    }
                                });
                            }
                        }
                    }
                })
            } else {
                match cand.result() {
                    // We only add an ambiguity cause if the goal would otherwise
                    // result in an error.
                    //
                    // FIXME: While this matches the behavior of the
                    // old solver, it is not the only way in which the unknowable
                    // candidates *weaken* coherence, they can also force otherwise
                    // sucessful normalization to be ambiguous.
                    Ok(Certainty::Maybe(_) | Certainty::Yes) => {
                        ambiguity_cause = None;
                        break;
                    }
                    Err(NoSolution) => continue,
                }
            }
        }

        if let Some(ambiguity_cause) = ambiguity_cause {
            self.causes.insert(ambiguity_cause);
        }

        ControlFlow::Continue(())
    }
}

fn search_ambiguity_causes<'tcx>(
    infcx: &InferCtxt<'tcx>,
    goal: Goal<'tcx, ty::Predicate<'tcx>>,
    causes: &mut FxIndexSet<IntercrateAmbiguityCause<'tcx>>,
) {
    infcx.visit_proof_tree(goal, &mut AmbiguityCausesVisitor { causes });
}
