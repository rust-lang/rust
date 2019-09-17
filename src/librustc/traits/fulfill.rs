use crate::infer::{InferCtxt, ShallowResolver};
use crate::mir::interpret::{GlobalId, ErrorHandled};
use crate::ty::{self, Ty, TypeFoldable, ToPolyTraitRef};
use crate::ty::error::ExpectedFound;
use rustc_data_structures::obligation_forest::{DoCompleted, Error, ForestObligation};
use rustc_data_structures::obligation_forest::{ObligationForest, ObligationProcessor};
use rustc_data_structures::obligation_forest::{ProcessResult};
use std::marker::PhantomData;

use super::CodeAmbiguity;
use super::CodeProjectionError;
use super::CodeSelectionError;
use super::engine::{TraitEngine, TraitEngineExt};
use super::{FulfillmentError, FulfillmentErrorCode};
use super::{ObligationCause, PredicateObligation};
use super::project;
use super::select::SelectionContext;
use super::{Unimplemented, ConstEvalFailure};

impl<'tcx> ForestObligation for PendingPredicateObligation<'tcx> {
    type Predicate = ty::Predicate<'tcx>;

    fn as_predicate(&self) -> &Self::Predicate { &self.obligation.predicate }
}

/// The fulfillment context is used to drive trait resolution. It
/// consists of a list of obligations that must be (eventually)
/// satisfied. The job is to track which are satisfied, which yielded
/// errors, and which are still pending. At any point, users can call
/// `select_where_possible`, and the fulfillment context will try to do
/// selection, retaining only those obligations that remain
/// ambiguous. This may be helpful in pushing type inference
/// along. Once all type inference constraints have been generated, the
/// method `select_all_or_error` can be used to report any remaining
/// ambiguous cases as errors.
pub struct FulfillmentContext<'tcx> {
    // A list of all obligations that have been registered with this
    // fulfillment context.
    predicates: ObligationForest<PendingPredicateObligation<'tcx>>,
    // Should this fulfillment context register type-lives-for-region
    // obligations on its parent infcx? In some cases, region
    // obligations are either already known to hold (normalization) or
    // hopefully verifed elsewhere (type-impls-bound), and therefore
    // should not be checked.
    //
    // Note that if we are normalizing a type that we already
    // know is well-formed, there should be no harm setting this
    // to true - all the region variables should be determinable
    // using the RFC 447 rules, which don't depend on
    // type-lives-for-region constraints, and because the type
    // is well-formed, the constraints should hold.
    register_region_obligations: bool,
    // Is it OK to register obligations into this infcx inside
    // an infcx snapshot?
    //
    // The "primary fulfillment" in many cases in typeck lives
    // outside of any snapshot, so any use of it inside a snapshot
    // will lead to trouble and therefore is checked against, but
    // other fulfillment contexts sometimes do live inside of
    // a snapshot (they don't *straddle* a snapshot, so there
    // is no trouble there).
    usable_in_snapshot: bool
}

#[derive(Clone, Debug)]
pub struct PendingPredicateObligation<'tcx> {
    pub obligation: PredicateObligation<'tcx>,
    pub stalled_on: Vec<Ty<'tcx>>,
}

// `PendingPredicateObligation` is used a lot. Make sure it doesn't unintentionally get bigger.
#[cfg(target_arch = "x86_64")]
static_assert_size!(PendingPredicateObligation<'_>, 136);

impl<'a, 'tcx> FulfillmentContext<'tcx> {
    /// Creates a new fulfillment context.
    pub fn new() -> FulfillmentContext<'tcx> {
        FulfillmentContext {
            predicates: ObligationForest::new(),
            register_region_obligations: true,
            usable_in_snapshot: false,
        }
    }

    pub fn new_in_snapshot() -> FulfillmentContext<'tcx> {
        FulfillmentContext {
            predicates: ObligationForest::new(),
            register_region_obligations: true,
            usable_in_snapshot: true,
        }
    }

    pub fn new_ignoring_regions() -> FulfillmentContext<'tcx> {
        FulfillmentContext {
            predicates: ObligationForest::new(),
            register_region_obligations: false,
            usable_in_snapshot: false
        }
    }

    /// Attempts to select obligations using `selcx`.
    fn select(
        &mut self,
        selcx: &mut SelectionContext<'a, 'tcx>,
    ) -> Result<(), Vec<FulfillmentError<'tcx>>> {
        debug!("select(obligation-forest-size={})", self.predicates.len());

        let mut errors = Vec::new();

        loop {
            debug!("select: starting another iteration");

            // Process pending obligations.
            let outcome = self.predicates.process_obligations(&mut FulfillProcessor {
                selcx,
                register_region_obligations: self.register_region_obligations
            }, DoCompleted::No);
            debug!("select: outcome={:#?}", outcome);

            // FIXME: if we kept the original cache key, we could mark projection
            // obligations as complete for the projection cache here.

            errors.extend(
                outcome.errors.into_iter()
                              .map(|e| to_fulfillment_error(e)));

            // If nothing new was added, no need to keep looping.
            if outcome.stalled {
                break;
            }
        }

        debug!("select({} predicates remaining, {} errors) done",
               self.predicates.len(), errors.len());

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

impl<'tcx> TraitEngine<'tcx> for FulfillmentContext<'tcx> {
    /// "Normalize" a projection type `<SomeType as SomeTrait>::X` by
    /// creating a fresh type variable `$0` as well as a projection
    /// predicate `<SomeType as SomeTrait>::X == $0`. When the
    /// inference engine runs, it will attempt to find an impl of
    /// `SomeTrait` or a where-clause that lets us unify `$0` with
    /// something concrete. If this fails, we'll unify `$0` with
    /// `projection_ty` again.
    fn normalize_projection_type(
        &mut self,
        infcx: &InferCtxt<'_, 'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        projection_ty: ty::ProjectionTy<'tcx>,
        cause: ObligationCause<'tcx>,
    ) -> Ty<'tcx> {
        debug!("normalize_projection_type(projection_ty={:?})",
               projection_ty);

        debug_assert!(!projection_ty.has_escaping_bound_vars());

        // FIXME(#20304) -- cache

        let mut selcx = SelectionContext::new(infcx);
        let mut obligations = vec![];
        let normalized_ty = project::normalize_projection_type(&mut selcx,
                                                               param_env,
                                                               projection_ty,
                                                               cause,
                                                               0,
                                                               &mut obligations);
        self.register_predicate_obligations(infcx, obligations);

        debug!("normalize_projection_type: result={:?}", normalized_ty);

        normalized_ty
    }

    fn register_predicate_obligation(
        &mut self,
        infcx: &InferCtxt<'_, 'tcx>,
        obligation: PredicateObligation<'tcx>,
    ) {
        // this helps to reduce duplicate errors, as well as making
        // debug output much nicer to read and so on.
        let obligation = infcx.resolve_vars_if_possible(&obligation);

        debug!("register_predicate_obligation(obligation={:?})", obligation);

        assert!(!infcx.is_in_snapshot() || self.usable_in_snapshot);

        self.predicates.register_obligation(PendingPredicateObligation {
            obligation,
            stalled_on: vec![]
        });
    }

    fn select_all_or_error(
        &mut self,
        infcx: &InferCtxt<'_, 'tcx>,
    ) -> Result<(), Vec<FulfillmentError<'tcx>>> {
        self.select_where_possible(infcx)?;

        let errors: Vec<_> =
            self.predicates.to_errors(CodeAmbiguity)
                           .into_iter()
                           .map(|e| to_fulfillment_error(e))
                           .collect();
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    fn select_where_possible(
        &mut self,
        infcx: &InferCtxt<'_, 'tcx>,
    ) -> Result<(), Vec<FulfillmentError<'tcx>>> {
        let mut selcx = SelectionContext::new(infcx);
        self.select(&mut selcx)
    }

    fn pending_obligations(&self) -> Vec<PredicateObligation<'tcx>> {
        self.predicates.map_pending_obligations(|o| o.obligation.clone())
    }
}

struct FulfillProcessor<'a, 'b, 'tcx> {
    selcx: &'a mut SelectionContext<'b, 'tcx>,
    register_region_obligations: bool,
}

fn mk_pending(os: Vec<PredicateObligation<'tcx>>) -> Vec<PendingPredicateObligation<'tcx>> {
    os.into_iter().map(|o| PendingPredicateObligation {
        obligation: o,
        stalled_on: vec![]
    }).collect()
}

impl<'a, 'b, 'tcx> ObligationProcessor for FulfillProcessor<'a, 'b, 'tcx> {
    type Obligation = PendingPredicateObligation<'tcx>;
    type Error = FulfillmentErrorCode<'tcx>;

    /// Processes a predicate obligation and returns either:
    /// - `Changed(v)` if the predicate is true, presuming that `v` are also true
    /// - `Unchanged` if we don't have enough info to be sure
    /// - `Error(e)` if the predicate does not hold
    ///
    /// This is always inlined, despite its size, because it has a single
    /// callsite and it is called *very* frequently.
    #[inline(always)]
    fn process_obligation(
        &mut self,
        pending_obligation: &mut Self::Obligation,
    ) -> ProcessResult<Self::Obligation, Self::Error> {
        // If we were stalled on some unresolved variables, first check
        // whether any of them have been resolved; if not, don't bother
        // doing more work yet
        if !pending_obligation.stalled_on.is_empty() {
            let mut changed = false;
            // This `for` loop was once a call to `all()`, but this lower-level
            // form was a perf win. See #64545 for details.
            for &ty in &pending_obligation.stalled_on {
                if ShallowResolver::new(self.selcx.infcx()).shallow_resolve_changed(ty) {
                    changed = true;
                    break;
                }
            }
            if !changed {
                debug!("process_predicate: pending obligation {:?} still stalled on {:?}",
                       self.selcx.infcx()
                           .resolve_vars_if_possible(&pending_obligation.obligation),
                       pending_obligation.stalled_on);
                return ProcessResult::Unchanged;
            }
            pending_obligation.stalled_on = vec![];
        }

        let obligation = &mut pending_obligation.obligation;

        if obligation.predicate.has_infer_types() {
            obligation.predicate =
                self.selcx.infcx().resolve_vars_if_possible(&obligation.predicate);
        }

        debug!("process_obligation: obligation = {:?} cause = {:?}", obligation, obligation.cause);

        match obligation.predicate {
            ty::Predicate::Trait(ref data) => {
                let trait_obligation = obligation.with(data.clone());

                if data.is_global() {
                    // no type variables present, can use evaluation for better caching.
                    // FIXME: consider caching errors too.
                    if self.selcx.infcx().predicate_must_hold_considering_regions(&obligation) {
                        debug!("selecting trait `{:?}` at depth {} evaluated to holds",
                               data, obligation.recursion_depth);
                        return ProcessResult::Changed(vec![])
                    }
                }

                match self.selcx.select(&trait_obligation) {
                    Ok(Some(vtable)) => {
                        debug!("selecting trait `{:?}` at depth {} yielded Ok(Some)",
                               data, obligation.recursion_depth);
                        ProcessResult::Changed(mk_pending(vtable.nested_obligations()))
                    }
                    Ok(None) => {
                        debug!("selecting trait `{:?}` at depth {} yielded Ok(None)",
                               data, obligation.recursion_depth);

                        // This is a bit subtle: for the most part, the
                        // only reason we can fail to make progress on
                        // trait selection is because we don't have enough
                        // information about the types in the trait. One
                        // exception is that we sometimes haven't decided
                        // what kind of closure a closure is. *But*, in
                        // that case, it turns out, the type of the
                        // closure will also change, because the closure
                        // also includes references to its upvars as part
                        // of its type, and those types are resolved at
                        // the same time.
                        //
                        // FIXME(#32286) logic seems false if no upvars
                        pending_obligation.stalled_on =
                            trait_ref_type_vars(self.selcx, data.to_poly_trait_ref());

                        debug!("process_predicate: pending obligation {:?} now stalled on {:?}",
                               self.selcx.infcx().resolve_vars_if_possible(obligation),
                               pending_obligation.stalled_on);

                        ProcessResult::Unchanged
                    }
                    Err(selection_err) => {
                        info!("selecting trait `{:?}` at depth {} yielded Err",
                              data, obligation.recursion_depth);

                        ProcessResult::Error(CodeSelectionError(selection_err))
                    }
                }
            }

            ty::Predicate::RegionOutlives(ref binder) => {
                match self.selcx.infcx().region_outlives_predicate(&obligation.cause, binder) {
                    Ok(()) => ProcessResult::Changed(vec![]),
                    Err(_) => ProcessResult::Error(CodeSelectionError(Unimplemented)),
                }
            }

            ty::Predicate::TypeOutlives(ref binder) => {
                // Check if there are higher-ranked vars.
                match binder.no_bound_vars() {
                    // If there are, inspect the underlying type further.
                    None => {
                        // Convert from `Binder<OutlivesPredicate<Ty, Region>>` to `Binder<Ty>`.
                        let binder = binder.map_bound_ref(|pred| pred.0);

                        // Check if the type has any bound vars.
                        match binder.no_bound_vars() {
                            // If so, this obligation is an error (for now). Eventually we should be
                            // able to support additional cases here, like `for<'a> &'a str: 'a`.
                            // NOTE: this is duplicate-implemented between here and fulfillment.
                            None => {
                                ProcessResult::Error(CodeSelectionError(Unimplemented))
                            }
                            // Otherwise, we have something of the form
                            // `for<'a> T: 'a where 'a not in T`, which we can treat as
                            // `T: 'static`.
                            Some(t_a) => {
                                let r_static = self.selcx.tcx().lifetimes.re_static;
                                if self.register_region_obligations {
                                    self.selcx.infcx().register_region_obligation_with_cause(
                                        t_a,
                                        r_static,
                                        &obligation.cause,
                                    );
                                }
                                ProcessResult::Changed(vec![])
                            }
                        }
                    }
                    // If there aren't, register the obligation.
                    Some(ty::OutlivesPredicate(t_a, r_b)) => {
                        if self.register_region_obligations {
                            self.selcx.infcx().register_region_obligation_with_cause(
                                t_a,
                                r_b,
                                &obligation.cause,
                            );
                        }
                        ProcessResult::Changed(vec![])
                    }
                }
            }

            ty::Predicate::Projection(ref data) => {
                let project_obligation = obligation.with(data.clone());
                match project::poly_project_and_unify_type(self.selcx, &project_obligation) {
                    Ok(None) => {
                        let tcx = self.selcx.tcx();
                        pending_obligation.stalled_on =
                            trait_ref_type_vars(self.selcx, data.to_poly_trait_ref(tcx));
                        ProcessResult::Unchanged
                    }
                    Ok(Some(os)) => ProcessResult::Changed(mk_pending(os)),
                    Err(e) => ProcessResult::Error(CodeProjectionError(e))
                }
            }

            ty::Predicate::ObjectSafe(trait_def_id) => {
                if !self.selcx.tcx().is_object_safe(trait_def_id) {
                    ProcessResult::Error(CodeSelectionError(Unimplemented))
                } else {
                    ProcessResult::Changed(vec![])
                }
            }

            ty::Predicate::ClosureKind(closure_def_id, closure_substs, kind) => {
                match self.selcx.infcx().closure_kind(closure_def_id, closure_substs) {
                    Some(closure_kind) => {
                        if closure_kind.extends(kind) {
                            ProcessResult::Changed(vec![])
                        } else {
                            ProcessResult::Error(CodeSelectionError(Unimplemented))
                        }
                    }
                    None => {
                        ProcessResult::Unchanged
                    }
                }
            }

            ty::Predicate::WellFormed(ty) => {
                match ty::wf::obligations(
                    self.selcx.infcx(),
                    obligation.param_env,
                    obligation.cause.body_id,
                    ty,
                    obligation.cause.span,
                ) {
                    None => {
                        pending_obligation.stalled_on = vec![ty];
                        ProcessResult::Unchanged
                    }
                    Some(os) => ProcessResult::Changed(mk_pending(os))
                }
            }

            ty::Predicate::Subtype(ref subtype) => {
                match self.selcx.infcx().subtype_predicate(&obligation.cause,
                                                           obligation.param_env,
                                                           subtype) {
                    None => {
                        // None means that both are unresolved.
                        pending_obligation.stalled_on = vec![subtype.skip_binder().a,
                                                             subtype.skip_binder().b];
                        ProcessResult::Unchanged
                    }
                    Some(Ok(ok)) => {
                        ProcessResult::Changed(mk_pending(ok.obligations))
                    }
                    Some(Err(err)) => {
                        let expected_found = ExpectedFound::new(subtype.skip_binder().a_is_expected,
                                                                subtype.skip_binder().a,
                                                                subtype.skip_binder().b);
                        ProcessResult::Error(
                            FulfillmentErrorCode::CodeSubtypeError(expected_found, err))
                    }
                }
            }

            ty::Predicate::ConstEvaluatable(def_id, substs) => {
                if obligation.param_env.has_local_value() {
                        ProcessResult::Unchanged
                } else {
                    if !substs.has_local_value() {
                        let instance = ty::Instance::resolve(
                            self.selcx.tcx().global_tcx(),
                            obligation.param_env,
                            def_id,
                            substs,
                        );
                        if let Some(instance) = instance {
                            let cid = GlobalId {
                                instance,
                                promoted: None,
                            };
                            match self.selcx.tcx().at(obligation.cause.span)
                                                    .const_eval(obligation.param_env.and(cid)) {
                                Ok(_) => ProcessResult::Changed(vec![]),
                                Err(err) => ProcessResult::Error(
                                    CodeSelectionError(ConstEvalFailure(err)))
                            }
                        } else {
                            ProcessResult::Error(CodeSelectionError(
                                ConstEvalFailure(ErrorHandled::TooGeneric)
                            ))
                        }
                    } else {
                        pending_obligation.stalled_on = substs.types().collect();
                        ProcessResult::Unchanged
                    }
                }
            }
        }
    }

    fn process_backedge<'c, I>(&mut self, cycle: I,
                               _marker: PhantomData<&'c PendingPredicateObligation<'tcx>>)
        where I: Clone + Iterator<Item=&'c PendingPredicateObligation<'tcx>>,
    {
        if self.selcx.coinductive_match(cycle.clone().map(|s| s.obligation.predicate)) {
            debug!("process_child_obligations: coinductive match");
        } else {
            let cycle: Vec<_> = cycle.map(|c| c.obligation.clone()).collect();
            self.selcx.infcx().report_overflow_error_cycle(&cycle);
        }
    }
}

/// Returns the set of type variables contained in a trait ref
fn trait_ref_type_vars<'a, 'tcx>(
    selcx: &mut SelectionContext<'a, 'tcx>,
    t: ty::PolyTraitRef<'tcx>,
) -> Vec<Ty<'tcx>> {
    t.skip_binder() // ok b/c this check doesn't care about regions
     .input_types()
     .map(|t| selcx.infcx().resolve_vars_if_possible(&t))
     .filter(|t| t.has_infer_types())
     .flat_map(|t| t.walk())
     .filter(|t| match t.sty { ty::Infer(_) => true, _ => false })
     .collect()
}

fn to_fulfillment_error<'tcx>(
    error: Error<PendingPredicateObligation<'tcx>, FulfillmentErrorCode<'tcx>>)
    -> FulfillmentError<'tcx>
{
    let obligation = error.backtrace.into_iter().next().unwrap().obligation;
    FulfillmentError::new(obligation, error.error)
}
