// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use infer::{RegionObligation, InferCtxt, InferOk};
use ty::{self, Ty, TypeFoldable, ToPolyTraitRef, ToPredicate};
use ty::error::ExpectedFound;
use rustc_data_structures::obligation_forest::{ObligationForest, Error};
use rustc_data_structures::obligation_forest::{ForestObligation, ObligationProcessor};
use std::marker::PhantomData;
use hir::def_id::DefId;

use super::CodeAmbiguity;
use super::CodeProjectionError;
use super::CodeSelectionError;
use super::{FulfillmentError, FulfillmentErrorCode};
use super::{ObligationCause, PredicateObligation, Obligation};
use super::project;
use super::select::SelectionContext;
use super::{Unimplemented, ConstEvalFailure};

impl<'tcx> ForestObligation for PendingPredicateObligation<'tcx> {
    type Predicate = ty::Predicate<'tcx>;

    fn as_predicate(&self) -> &Self::Predicate { &self.obligation.predicate }
}

/// The fulfillment context is used to drive trait resolution.  It
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
}

#[derive(Clone, Debug)]
pub struct PendingPredicateObligation<'tcx> {
    pub obligation: PredicateObligation<'tcx>,
    pub stalled_on: Vec<Ty<'tcx>>,
}

impl<'a, 'gcx, 'tcx> FulfillmentContext<'tcx> {
    /// Creates a new fulfillment context.
    pub fn new() -> FulfillmentContext<'tcx> {
        FulfillmentContext {
            predicates: ObligationForest::new(),
            register_region_obligations: true
        }
    }

    pub fn new_ignoring_regions() -> FulfillmentContext<'tcx> {
        FulfillmentContext {
            predicates: ObligationForest::new(),
            register_region_obligations: false
        }
    }

    /// "Normalize" a projection type `<SomeType as SomeTrait>::X` by
    /// creating a fresh type variable `$0` as well as a projection
    /// predicate `<SomeType as SomeTrait>::X == $0`. When the
    /// inference engine runs, it will attempt to find an impl of
    /// `SomeTrait` or a where clause that lets us unify `$0` with
    /// something concrete. If this fails, we'll unify `$0` with
    /// `projection_ty` again.
    pub fn normalize_projection_type(&mut self,
                                     infcx: &InferCtxt<'a, 'gcx, 'tcx>,
                                     param_env: ty::ParamEnv<'tcx>,
                                     projection_ty: ty::ProjectionTy<'tcx>,
                                     cause: ObligationCause<'tcx>)
                                     -> Ty<'tcx>
    {
        debug!("normalize_projection_type(projection_ty={:?})",
               projection_ty);

        assert!(!projection_ty.has_escaping_regions());

        // FIXME(#20304) -- cache

        let mut selcx = SelectionContext::new(infcx);
        let normalized = project::normalize_projection_type(&mut selcx,
                                                            param_env,
                                                            projection_ty,
                                                            cause,
                                                            0);

        for obligation in normalized.obligations {
            self.register_predicate_obligation(infcx, obligation);
        }

        debug!("normalize_projection_type: result={:?}", normalized.value);

        normalized.value
    }

    /// Requires that `ty` must implement the trait with `def_id` in
    /// the given environment. This trait must not have any type
    /// parameters (except for `Self`).
    pub fn register_bound(&mut self,
                          infcx: &InferCtxt<'a, 'gcx, 'tcx>,
                          param_env: ty::ParamEnv<'tcx>,
                          ty: Ty<'tcx>,
                          def_id: DefId,
                          cause: ObligationCause<'tcx>)
    {
        let trait_ref = ty::TraitRef {
            def_id,
            substs: infcx.tcx.mk_substs_trait(ty, &[]),
        };
        self.register_predicate_obligation(infcx, Obligation {
            cause,
            recursion_depth: 0,
            param_env,
            predicate: trait_ref.to_predicate()
        });
    }

    pub fn register_predicate_obligation(&mut self,
                                         infcx: &InferCtxt<'a, 'gcx, 'tcx>,
                                         obligation: PredicateObligation<'tcx>)
    {
        // this helps to reduce duplicate errors, as well as making
        // debug output much nicer to read and so on.
        let obligation = infcx.resolve_type_vars_if_possible(&obligation);

        debug!("register_predicate_obligation(obligation={:?})", obligation);

        assert!(!infcx.is_in_snapshot());

        self.predicates.register_obligation(PendingPredicateObligation {
            obligation,
            stalled_on: vec![]
        });
    }

    pub fn register_predicate_obligations<I>(&mut self,
                                             infcx: &InferCtxt<'a, 'gcx, 'tcx>,
                                             obligations: I)
        where I: IntoIterator<Item = PredicateObligation<'tcx>>
    {
        for obligation in obligations {
            self.register_predicate_obligation(infcx, obligation);
        }
    }

    pub fn select_all_or_error(&mut self,
                               infcx: &InferCtxt<'a, 'gcx, 'tcx>)
                               -> Result<(),Vec<FulfillmentError<'tcx>>>
    {
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

    pub fn select_where_possible(&mut self,
                                 infcx: &InferCtxt<'a, 'gcx, 'tcx>)
                                 -> Result<(),Vec<FulfillmentError<'tcx>>>
    {
        let mut selcx = SelectionContext::new(infcx);
        self.select(&mut selcx)
    }

    pub fn pending_obligations(&self) -> Vec<PendingPredicateObligation<'tcx>> {
        self.predicates.pending_obligations()
    }

    /// Attempts to select obligations using `selcx`. If `only_new_obligations` is true, then it
    /// only attempts to select obligations that haven't been seen before.
    fn select(&mut self, selcx: &mut SelectionContext<'a, 'gcx, 'tcx>)
              -> Result<(),Vec<FulfillmentError<'tcx>>> {
        debug!("select(obligation-forest-size={})", self.predicates.len());

        let mut errors = Vec::new();

        loop {
            debug!("select: starting another iteration");

            // Process pending obligations.
            let outcome = self.predicates.process_obligations(&mut FulfillProcessor {
                selcx,
                register_region_obligations: self.register_region_obligations
            });
            debug!("select: outcome={:?}", outcome);

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

struct FulfillProcessor<'a, 'b: 'a, 'gcx: 'tcx, 'tcx: 'b> {
    selcx: &'a mut SelectionContext<'b, 'gcx, 'tcx>,
    register_region_obligations: bool
}

impl<'a, 'b, 'gcx, 'tcx> ObligationProcessor for FulfillProcessor<'a, 'b, 'gcx, 'tcx> {
    type Obligation = PendingPredicateObligation<'tcx>;
    type Error = FulfillmentErrorCode<'tcx>;

    fn process_obligation(&mut self,
                          obligation: &mut Self::Obligation)
                          -> Result<Option<Vec<Self::Obligation>>, Self::Error>
    {
        process_predicate(self.selcx, obligation, self.register_region_obligations)
            .map(|os| os.map(|os| os.into_iter().map(|o| PendingPredicateObligation {
                obligation: o,
                stalled_on: vec![]
            }).collect()))
    }

    fn process_backedge<'c, I>(&mut self, cycle: I,
                               _marker: PhantomData<&'c PendingPredicateObligation<'tcx>>)
        where I: Clone + Iterator<Item=&'c PendingPredicateObligation<'tcx>>,
    {
        if self.selcx.coinductive_match(cycle.clone().map(|s| s.obligation.predicate)) {
            debug!("process_child_obligations: coinductive match");
        } else {
            let cycle : Vec<_> = cycle.map(|c| c.obligation.clone()).collect();
            self.selcx.infcx().report_overflow_error_cycle(&cycle);
        }
    }
}

/// Return the set of type variables contained in a trait ref
fn trait_ref_type_vars<'a, 'gcx, 'tcx>(selcx: &mut SelectionContext<'a, 'gcx, 'tcx>,
                                       t: ty::PolyTraitRef<'tcx>) -> Vec<Ty<'tcx>>
{
    t.skip_binder() // ok b/c this check doesn't care about regions
     .input_types()
     .map(|t| selcx.infcx().resolve_type_vars_if_possible(&t))
     .filter(|t| t.has_infer_types())
     .flat_map(|t| t.walk())
     .filter(|t| match t.sty { ty::TyInfer(_) => true, _ => false })
     .collect()
}

/// Processes a predicate obligation and returns either:
/// - `Ok(Some(v))` if the predicate is true, presuming that `v` are also true
/// - `Ok(None)` if we don't have enough info to be sure
/// - `Err` if the predicate does not hold
fn process_predicate<'a, 'gcx, 'tcx>(
    selcx: &mut SelectionContext<'a, 'gcx, 'tcx>,
    pending_obligation: &mut PendingPredicateObligation<'tcx>,
    register_region_obligations: bool)
    -> Result<Option<Vec<PredicateObligation<'tcx>>>,
              FulfillmentErrorCode<'tcx>>
{
    // if we were stalled on some unresolved variables, first check
    // whether any of them have been resolved; if not, don't bother
    // doing more work yet
    if !pending_obligation.stalled_on.is_empty() {
        if pending_obligation.stalled_on.iter().all(|&ty| {
            let resolved_ty = selcx.infcx().shallow_resolve(&ty);
            resolved_ty == ty // nothing changed here
        }) {
            debug!("process_predicate: pending obligation {:?} still stalled on {:?}",
                   selcx.infcx().resolve_type_vars_if_possible(&pending_obligation.obligation),
                   pending_obligation.stalled_on);
            return Ok(None);
        }
        pending_obligation.stalled_on = vec![];
    }

    let obligation = &mut pending_obligation.obligation;

    if obligation.predicate.has_infer_types() {
        obligation.predicate = selcx.infcx().resolve_type_vars_if_possible(&obligation.predicate);
    }

    match obligation.predicate {
        ty::Predicate::Trait(ref data) => {
            let trait_obligation = obligation.with(data.clone());

            if data.is_global() {
                // no type variables present, can use evaluation for better caching.
                // FIXME: consider caching errors too.
                if
                    // make defaulted unit go through the slow path for better warnings,
                    // please remove this when the warnings are removed.
                    !trait_obligation.predicate.skip_binder().self_ty().is_defaulted_unit() &&
                    selcx.evaluate_obligation_conservatively(&obligation) {
                    debug!("selecting trait `{:?}` at depth {} evaluated to holds",
                           data, obligation.recursion_depth);
                    return Ok(Some(vec![]))
                }
            }

            match selcx.select(&trait_obligation) {
                Ok(Some(vtable)) => {
                    debug!("selecting trait `{:?}` at depth {} yielded Ok(Some)",
                           data, obligation.recursion_depth);
                    Ok(Some(vtable.nested_obligations()))
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
                        trait_ref_type_vars(selcx, data.to_poly_trait_ref());

                    debug!("process_predicate: pending obligation {:?} now stalled on {:?}",
                           selcx.infcx().resolve_type_vars_if_possible(obligation),
                           pending_obligation.stalled_on);

                    Ok(None)
                }
                Err(selection_err) => {
                    info!("selecting trait `{:?}` at depth {} yielded Err",
                          data, obligation.recursion_depth);

                    Err(CodeSelectionError(selection_err))
                }
            }
        }

        ty::Predicate::Equate(ref binder) => {
            match selcx.infcx().equality_predicate(&obligation.cause,
                                                   obligation.param_env,
                                                   binder) {
                Ok(InferOk { obligations, value: () }) => {
                    Ok(Some(obligations))
                },
                Err(_) => Err(CodeSelectionError(Unimplemented)),
            }
        }

        ty::Predicate::RegionOutlives(ref binder) => {
            match selcx.infcx().region_outlives_predicate(&obligation.cause, binder) {
                Ok(()) => Ok(Some(Vec::new())),
                Err(_) => Err(CodeSelectionError(Unimplemented)),
            }
        }

        ty::Predicate::TypeOutlives(ref binder) => {
            // Check if there are higher-ranked regions.
            match binder.no_late_bound_regions() {
                // If there are, inspect the underlying type further.
                None => {
                    // Convert from `Binder<OutlivesPredicate<Ty, Region>>` to `Binder<Ty>`.
                    let binder = binder.map_bound_ref(|pred| pred.0);

                    // Check if the type has any bound regions.
                    match binder.no_late_bound_regions() {
                        // If so, this obligation is an error (for now). Eventually we should be
                        // able to support additional cases here, like `for<'a> &'a str: 'a`.
                        None => {
                            Err(CodeSelectionError(Unimplemented))
                        }
                        // Otherwise, we have something of the form
                        // `for<'a> T: 'a where 'a not in T`, which we can treat as `T: 'static`.
                        Some(t_a) => {
                            let r_static = selcx.tcx().types.re_static;
                            if register_region_obligations {
                                selcx.infcx().register_region_obligation(
                                    obligation.cause.body_id,
                                    RegionObligation {
                                        sup_type: t_a,
                                        sub_region: r_static,
                                        cause: obligation.cause.clone(),
                                    });
                            }
                            Ok(Some(vec![]))
                        }
                    }
                }
                // If there aren't, register the obligation.
                Some(ty::OutlivesPredicate(t_a, r_b)) => {
                    if register_region_obligations {
                        selcx.infcx().register_region_obligation(
                            obligation.cause.body_id,
                            RegionObligation {
                                sup_type: t_a,
                                sub_region: r_b,
                                cause: obligation.cause.clone()
                            });
                    }
                    Ok(Some(vec![]))
                }
            }
        }

        ty::Predicate::Projection(ref data) => {
            let project_obligation = obligation.with(data.clone());
            match project::poly_project_and_unify_type(selcx, &project_obligation) {
                Ok(None) => {
                    let tcx = selcx.tcx();
                    pending_obligation.stalled_on =
                        trait_ref_type_vars(selcx, data.to_poly_trait_ref(tcx));
                    Ok(None)
                }
                Ok(v) => Ok(v),
                Err(e) => Err(CodeProjectionError(e))
            }
        }

        ty::Predicate::ObjectSafe(trait_def_id) => {
            if !selcx.tcx().is_object_safe(trait_def_id) {
                Err(CodeSelectionError(Unimplemented))
            } else {
                Ok(Some(Vec::new()))
            }
        }

        ty::Predicate::ClosureKind(closure_def_id, closure_substs, kind) => {
            match selcx.infcx().closure_kind(closure_def_id, closure_substs) {
                Some(closure_kind) => {
                    if closure_kind.extends(kind) {
                        Ok(Some(vec![]))
                    } else {
                        Err(CodeSelectionError(Unimplemented))
                    }
                }
                None => {
                    Ok(None)
                }
            }
        }

        ty::Predicate::WellFormed(ty) => {
            match ty::wf::obligations(selcx.infcx(),
                                      obligation.param_env,
                                      obligation.cause.body_id,
                                      ty, obligation.cause.span) {
                None => {
                    pending_obligation.stalled_on = vec![ty];
                    Ok(None)
                }
                s => Ok(s)
            }
        }

        ty::Predicate::Subtype(ref subtype) => {
            match selcx.infcx().subtype_predicate(&obligation.cause,
                                                  obligation.param_env,
                                                  subtype) {
                None => {
                    // none means that both are unresolved
                    pending_obligation.stalled_on = vec![subtype.skip_binder().a,
                                                         subtype.skip_binder().b];
                    Ok(None)
                }
                Some(Ok(ok)) => {
                    Ok(Some(ok.obligations))
                }
                Some(Err(err)) => {
                    let expected_found = ExpectedFound::new(subtype.skip_binder().a_is_expected,
                                                            subtype.skip_binder().a,
                                                            subtype.skip_binder().b);
                    Err(FulfillmentErrorCode::CodeSubtypeError(expected_found, err))
                }
            }
        }

        ty::Predicate::ConstEvaluatable(def_id, substs) => {
            match selcx.tcx().lift_to_global(&obligation.param_env) {
                None => {
                    Ok(None)
                }
                Some(param_env) => {
                    match selcx.tcx().lift_to_global(&substs) {
                        None => {
                            pending_obligation.stalled_on = substs.types().collect();
                            Ok(None)
                        }
                        Some(substs) => {
                            match selcx.tcx().at(obligation.cause.span)
                                             .const_eval(param_env.and((def_id, substs))) {
                                Ok(_) => Ok(Some(vec![])),
                                Err(e) => Err(CodeSelectionError(ConstEvalFailure(e)))
                            }
                        }
                    }
                }
            }
        }
    }
}

fn to_fulfillment_error<'tcx>(
    error: Error<PendingPredicateObligation<'tcx>, FulfillmentErrorCode<'tcx>>)
    -> FulfillmentError<'tcx>
{
    let obligation = error.backtrace.into_iter().next().unwrap().obligation;
    FulfillmentError::new(obligation, error.error)
}
