// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use dep_graph::DepGraph;
use infer::{InferCtxt, InferOk};
use ty::{self, Ty, TypeFoldable, ToPolyTraitRef, TyCtxt, ToPredicate};
use ty::subst::Subst;
use rustc_data_structures::obligation_forest::{ObligationForest, Error};
use rustc_data_structures::obligation_forest::{ForestObligation, ObligationProcessor};
use std::marker::PhantomData;
use std::mem;
use syntax::ast;
use util::nodemap::{FxHashSet, NodeMap};
use hir::def_id::DefId;

use super::CodeAmbiguity;
use super::CodeProjectionError;
use super::CodeSelectionError;
use super::{FulfillmentError, FulfillmentErrorCode, SelectionError};
use super::{ObligationCause, BuiltinDerivedObligation};
use super::{PredicateObligation, TraitObligation, Obligation};
use super::project;
use super::select::SelectionContext;
use super::Unimplemented;

impl<'tcx> ForestObligation for PendingPredicateObligation<'tcx> {
    type Predicate = ty::Predicate<'tcx>;

    fn as_predicate(&self) -> &Self::Predicate { &self.obligation.predicate }
}

pub struct GlobalFulfilledPredicates<'tcx> {
    set: FxHashSet<ty::PolyTraitPredicate<'tcx>>,
    dep_graph: DepGraph,
}

/// The fulfillment context is used to drive trait resolution.  It
/// consists of a list of obligations that must be (eventually)
/// satisfied. The job is to track which are satisfied, which yielded
/// errors, and which are still pending. At any point, users can call
/// `select_where_possible`, and the fulfilment context will try to do
/// selection, retaining only those obligations that remain
/// ambiguous. This may be helpful in pushing type inference
/// along. Once all type inference constraints have been generated, the
/// method `select_all_or_error` can be used to report any remaining
/// ambiguous cases as errors.

pub struct FulfillmentContext<'tcx> {
    // A list of all obligations that have been registered with this
    // fulfillment context.
    predicates: ObligationForest<PendingPredicateObligation<'tcx>>,

    // A set of constraints that regionck must validate. Each
    // constraint has the form `T:'a`, meaning "some type `T` must
    // outlive the lifetime 'a". These constraints derive from
    // instantiated type parameters. So if you had a struct defined
    // like
    //
    //     struct Foo<T:'static> { ... }
    //
    // then in some expression `let x = Foo { ... }` it will
    // instantiate the type parameter `T` with a fresh type `$0`. At
    // the same time, it will record a region obligation of
    // `$0:'static`. This will get checked later by regionck. (We
    // can't generally check these things right away because we have
    // to wait until types are resolved.)
    //
    // These are stored in a map keyed to the id of the innermost
    // enclosing fn body / static initializer expression. This is
    // because the location where the obligation was incurred can be
    // relevant with respect to which sublifetime assumptions are in
    // place. The reason that we store under the fn-id, and not
    // something more fine-grained, is so that it is easier for
    // regionck to be sure that it has found *all* the region
    // obligations (otherwise, it's easy to fail to walk to a
    // particular node-id).
    region_obligations: NodeMap<Vec<RegionObligation<'tcx>>>,

    // A list of obligations that need to be deferred to
    // a later time for them to be properly fulfilled.
    deferred_obligations: Vec<DeferredObligation<'tcx>>,
}

#[derive(Clone)]
pub struct RegionObligation<'tcx> {
    pub sub_region: &'tcx ty::Region,
    pub sup_type: Ty<'tcx>,
    pub cause: ObligationCause<'tcx>,
}

#[derive(Clone, Debug)]
pub struct PendingPredicateObligation<'tcx> {
    pub obligation: PredicateObligation<'tcx>,
    pub stalled_on: Vec<Ty<'tcx>>,
}

/// An obligation which cannot be fulfilled in the context
/// it was registered in, such as auto trait obligations on
/// `impl Trait`, which require the concrete type to be
/// available, only guaranteed after finishing type-checking.
#[derive(Clone, Debug)]
pub struct DeferredObligation<'tcx> {
    pub predicate: ty::PolyTraitPredicate<'tcx>,
    pub cause: ObligationCause<'tcx>
}

impl<'a, 'gcx, 'tcx> DeferredObligation<'tcx> {
    /// If possible, create a `DeferredObligation` from
    /// a trait predicate which had failed selection,
    /// but could succeed later.
    pub fn from_select_error(tcx: TyCtxt<'a, 'gcx, 'tcx>,
                             obligation: &TraitObligation<'tcx>,
                             selection_err: &SelectionError<'tcx>)
                             -> Option<DeferredObligation<'tcx>> {
        if let Unimplemented = *selection_err {
            if DeferredObligation::must_defer(tcx, &obligation.predicate) {
                return Some(DeferredObligation {
                    predicate: obligation.predicate.clone(),
                    cause: obligation.cause.clone()
                });
            }
        }

        None
    }

    /// Returns true if the given trait predicate can be
    /// fulfilled at a later time.
    pub fn must_defer(tcx: TyCtxt<'a, 'gcx, 'tcx>,
                      predicate: &ty::PolyTraitPredicate<'tcx>)
                      -> bool {
        // Auto trait obligations on `impl Trait`.
        if tcx.trait_has_default_impl(predicate.def_id()) {
            let substs = predicate.skip_binder().trait_ref.substs;
            if substs.types().count() == 1 && substs.regions().next().is_none() {
                if let ty::TyAnon(..) = predicate.skip_binder().self_ty().sty {
                    return true;
                }
            }
        }

        false
    }

    /// If possible, return the nested obligations required
    /// to fulfill this obligation.
    pub fn try_select(&self, tcx: TyCtxt<'a, 'gcx, 'tcx>)
                      -> Option<Vec<PredicateObligation<'tcx>>> {
        if let ty::TyAnon(def_id, substs) = self.predicate.skip_binder().self_ty().sty {
            let ty = if def_id.is_local() {
                tcx.item_types.borrow().get(&def_id).cloned()
            } else {
                Some(tcx.item_type(def_id))
            };
            // We can resolve the `impl Trait` to its concrete type.
            if let Some(concrete_ty) = ty.subst(tcx, substs) {
                let predicate = ty::TraitRef {
                    def_id: self.predicate.def_id(),
                    substs: tcx.mk_substs_trait(concrete_ty, &[])
                }.to_predicate();

                let original_obligation = Obligation::new(self.cause.clone(),
                                                          self.predicate.clone());
                let cause = original_obligation.derived_cause(BuiltinDerivedObligation);
                return Some(vec![Obligation::new(cause, predicate)]);
            }
        }

        None
    }

    /// Return the `PredicateObligation` this was created from.
    pub fn to_obligation(&self) -> PredicateObligation<'tcx> {
        let predicate = ty::Predicate::Trait(self.predicate.clone());
        Obligation::new(self.cause.clone(), predicate)
    }

    /// Return an error as if this obligation had failed.
    pub fn to_error(&self) -> FulfillmentError<'tcx> {
        FulfillmentError::new(self.to_obligation(), CodeSelectionError(Unimplemented))
    }
}

impl<'a, 'gcx, 'tcx> FulfillmentContext<'tcx> {
    /// Creates a new fulfillment context.
    pub fn new() -> FulfillmentContext<'tcx> {
        FulfillmentContext {
            predicates: ObligationForest::new(),
            region_obligations: NodeMap(),
            deferred_obligations: vec![],
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
                                     projection_ty: ty::ProjectionTy<'tcx>,
                                     cause: ObligationCause<'tcx>)
                                     -> Ty<'tcx>
    {
        debug!("normalize_projection_type(projection_ty={:?})",
               projection_ty);

        assert!(!projection_ty.has_escaping_regions());

        // FIXME(#20304) -- cache

        let mut selcx = SelectionContext::new(infcx);
        let normalized = project::normalize_projection_type(&mut selcx, projection_ty, cause, 0);

        for obligation in normalized.obligations {
            self.register_predicate_obligation(infcx, obligation);
        }

        debug!("normalize_projection_type: result={:?}", normalized.value);

        normalized.value
    }

    pub fn register_bound(&mut self,
                          infcx: &InferCtxt<'a, 'gcx, 'tcx>,
                          ty: Ty<'tcx>,
                          def_id: DefId,
                          cause: ObligationCause<'tcx>)
    {
        let trait_ref = ty::TraitRef {
            def_id: def_id,
            substs: infcx.tcx.mk_substs_trait(ty, &[]),
        };
        self.register_predicate_obligation(infcx, Obligation {
            cause: cause,
            recursion_depth: 0,
            predicate: trait_ref.to_predicate()
        });
    }

    pub fn register_region_obligation(&mut self,
                                      t_a: Ty<'tcx>,
                                      r_b: &'tcx ty::Region,
                                      cause: ObligationCause<'tcx>)
    {
        register_region_obligation(t_a, r_b, cause, &mut self.region_obligations);
    }

    pub fn register_predicate_obligation(&mut self,
                                         infcx: &InferCtxt<'a, 'gcx, 'tcx>,
                                         obligation: PredicateObligation<'tcx>)
    {
        // this helps to reduce duplicate errors, as well as making
        // debug output much nicer to read and so on.
        let obligation = infcx.resolve_type_vars_if_possible(&obligation);

        debug!("register_predicate_obligation(obligation={:?})", obligation);

        infcx.obligations_in_snapshot.set(true);

        if infcx.tcx.fulfilled_predicates.borrow().check_duplicate(&obligation.predicate) {
            debug!("register_predicate_obligation: duplicate");
            return
        }

        self.predicates.register_obligation(PendingPredicateObligation {
            obligation: obligation,
            stalled_on: vec![]
        });
    }

    pub fn region_obligations(&self,
                              body_id: ast::NodeId)
                              -> &[RegionObligation<'tcx>]
    {
        match self.region_obligations.get(&body_id) {
            None => Default::default(),
            Some(vec) => vec,
        }
    }

    pub fn select_all_or_error(&mut self,
                               infcx: &InferCtxt<'a, 'gcx, 'tcx>)
                               -> Result<(),Vec<FulfillmentError<'tcx>>>
    {
        self.select_where_possible(infcx)?;

        // Fail all of the deferred obligations that haven't
        // been otherwise removed from the context.
        let deferred_errors = self.deferred_obligations.iter()
                                  .map(|d| d.to_error());

        let errors: Vec<_> =
            self.predicates.to_errors(CodeAmbiguity)
                           .into_iter()
                           .map(|e| to_fulfillment_error(e))
                           .chain(deferred_errors)
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

    pub fn take_deferred_obligations(&mut self) -> Vec<DeferredObligation<'tcx>> {
        mem::replace(&mut self.deferred_obligations, vec![])
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
                selcx: selcx,
                region_obligations: &mut self.region_obligations,
                deferred_obligations: &mut self.deferred_obligations
            });
            debug!("select: outcome={:?}", outcome);

            // these are obligations that were proven to be true.
            for pending_obligation in outcome.completed {
                let predicate = &pending_obligation.obligation.predicate;
                selcx.tcx().fulfilled_predicates.borrow_mut()
                           .add_if_global(selcx.tcx(), predicate);
            }

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
    region_obligations: &'a mut NodeMap<Vec<RegionObligation<'tcx>>>,
    deferred_obligations: &'a mut Vec<DeferredObligation<'tcx>>
}

impl<'a, 'b, 'gcx, 'tcx> ObligationProcessor for FulfillProcessor<'a, 'b, 'gcx, 'tcx> {
    type Obligation = PendingPredicateObligation<'tcx>;
    type Error = FulfillmentErrorCode<'tcx>;

    fn process_obligation(&mut self,
                          obligation: &mut Self::Obligation)
                          -> Result<Option<Vec<Self::Obligation>>, Self::Error>
    {
        process_predicate(self.selcx,
                          obligation,
                          self.region_obligations,
                          self.deferred_obligations)
            .map(|os| os.map(|os| os.into_iter().map(|o| PendingPredicateObligation {
                obligation: o,
                stalled_on: vec![]
            }).collect()))
    }

    fn process_backedge<'c, I>(&mut self, cycle: I,
                               _marker: PhantomData<&'c PendingPredicateObligation<'tcx>>)
        where I: Clone + Iterator<Item=&'c PendingPredicateObligation<'tcx>>,
    {
        if coinductive_match(self.selcx, cycle.clone()) {
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
    region_obligations: &mut NodeMap<Vec<RegionObligation<'tcx>>>,
    deferred_obligations: &mut Vec<DeferredObligation<'tcx>>)
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
            if selcx.tcx().fulfilled_predicates.borrow().check_duplicate_trait(data) {
                return Ok(Some(vec![]));
            }

            let trait_obligation = obligation.with(data.clone());
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

                    let defer = DeferredObligation::from_select_error(selcx.tcx(),
                                                                      &trait_obligation,
                                                                      &selection_err);
                    if let Some(deferred_obligation) = defer {
                        if let Some(nested) = deferred_obligation.try_select(selcx.tcx()) {
                            Ok(Some(nested))
                        } else {
                            // Pretend that the obligation succeeded,
                            // but record it for later.
                            deferred_obligations.push(deferred_obligation);
                            Ok(Some(vec![]))
                        }
                    } else {
                        Err(CodeSelectionError(selection_err))
                    }
                }
            }
        }

        ty::Predicate::Equate(ref binder) => {
            match selcx.infcx().equality_predicate(&obligation.cause, binder) {
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
            match selcx.tcx().no_late_bound_regions(binder) {
                // If there are, inspect the underlying type further.
                None => {
                    // Convert from `Binder<OutlivesPredicate<Ty, Region>>` to `Binder<Ty>`.
                    let binder = binder.map_bound_ref(|pred| pred.0);

                    // Check if the type has any bound regions.
                    match selcx.tcx().no_late_bound_regions(&binder) {
                        // If so, this obligation is an error (for now). Eventually we should be
                        // able to support additional cases here, like `for<'a> &'a str: 'a`.
                        None => {
                            Err(CodeSelectionError(Unimplemented))
                        }
                        // Otherwise, we have something of the form
                        // `for<'a> T: 'a where 'a not in T`, which we can treat as `T: 'static`.
                        Some(t_a) => {
                            let r_static = selcx.tcx().mk_region(ty::ReStatic);
                            register_region_obligation(t_a, r_static,
                                                       obligation.cause.clone(),
                                                       region_obligations);
                            Ok(Some(vec![]))
                        }
                    }
                }
                // If there aren't, register the obligation.
                Some(ty::OutlivesPredicate(t_a, r_b)) => {
                    register_region_obligation(t_a, r_b,
                                               obligation.cause.clone(),
                                               region_obligations);
                    Ok(Some(vec![]))
                }
            }
        }

        ty::Predicate::Projection(ref data) => {
            let project_obligation = obligation.with(data.clone());
            match project::poly_project_and_unify_type(selcx, &project_obligation) {
                Ok(None) => {
                    pending_obligation.stalled_on =
                        trait_ref_type_vars(selcx, data.to_poly_trait_ref());
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

        ty::Predicate::ClosureKind(closure_def_id, kind) => {
            match selcx.infcx().closure_kind(closure_def_id) {
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
            match ty::wf::obligations(selcx.infcx(), obligation.cause.body_id,
                                      ty, obligation.cause.span) {
                None => {
                    pending_obligation.stalled_on = vec![ty];
                    Ok(None)
                }
                s => Ok(s)
            }
        }
    }
}

/// For defaulted traits, we use a co-inductive strategy to solve, so
/// that recursion is ok. This routine returns true if the top of the
/// stack (`cycle[0]`):
/// - is a defaulted trait, and
/// - it also appears in the backtrace at some position `X`; and,
/// - all the predicates at positions `X..` between `X` an the top are
///   also defaulted traits.
fn coinductive_match<'a,'c,'gcx,'tcx,I>(selcx: &mut SelectionContext<'a,'gcx,'tcx>,
                                        cycle: I) -> bool
    where I: Iterator<Item=&'c PendingPredicateObligation<'tcx>>,
          'tcx: 'c
{
    let mut cycle = cycle;
    cycle
        .all(|bt_obligation| {
            let result = coinductive_obligation(selcx, &bt_obligation.obligation);
            debug!("coinductive_match: bt_obligation={:?} coinductive={}",
                   bt_obligation, result);
            result
        })
}

fn coinductive_obligation<'a,'gcx,'tcx>(selcx: &SelectionContext<'a,'gcx,'tcx>,
                                          obligation: &PredicateObligation<'tcx>)
                                          -> bool {
    match obligation.predicate {
        ty::Predicate::Trait(ref data) => {
            selcx.tcx().trait_has_default_impl(data.def_id())
        }
        _ => {
            false
        }
    }
}

fn register_region_obligation<'tcx>(t_a: Ty<'tcx>,
                                    r_b: &'tcx ty::Region,
                                    cause: ObligationCause<'tcx>,
                                    region_obligations: &mut NodeMap<Vec<RegionObligation<'tcx>>>)
{
    let region_obligation = RegionObligation { sup_type: t_a,
                                               sub_region: r_b,
                                               cause: cause };

    debug!("register_region_obligation({:?}, cause={:?})",
           region_obligation, region_obligation.cause);

    region_obligations.entry(region_obligation.cause.body_id)
                      .or_insert(vec![])
                      .push(region_obligation);

}

impl<'a, 'gcx, 'tcx> GlobalFulfilledPredicates<'gcx> {
    pub fn new(dep_graph: DepGraph) -> GlobalFulfilledPredicates<'gcx> {
        GlobalFulfilledPredicates {
            set: FxHashSet(),
            dep_graph: dep_graph,
        }
    }

    pub fn check_duplicate(&self, key: &ty::Predicate<'tcx>) -> bool {
        if let ty::Predicate::Trait(ref data) = *key {
            self.check_duplicate_trait(data)
        } else {
            false
        }
    }

    pub fn check_duplicate_trait(&self, data: &ty::PolyTraitPredicate<'tcx>) -> bool {
        // For the global predicate registry, when we find a match, it
        // may have been computed by some other task, so we want to
        // add a read from the node corresponding to the predicate
        // processing to make sure we get the transitive dependencies.
        if self.set.contains(data) {
            debug_assert!(data.is_global());
            self.dep_graph.read(data.dep_node());
            debug!("check_duplicate: global predicate `{:?}` already proved elsewhere", data);

            true
        } else {
            false
        }
    }

    fn add_if_global(&mut self, tcx: TyCtxt<'a, 'gcx, 'tcx>, key: &ty::Predicate<'tcx>) {
        if let ty::Predicate::Trait(ref data) = *key {
            // We only add things to the global predicate registry
            // after the current task has proved them, and hence
            // already has the required read edges, so we don't need
            // to add any more edges here.
            if data.is_global() {
                // Don't cache predicates which were fulfilled
                // by deferring them for later fulfillment.
                if DeferredObligation::must_defer(tcx, data) {
                    return;
                }

                if let Some(data) = tcx.lift_to_global(data) {
                    if self.set.insert(data.clone()) {
                        debug!("add_if_global: global predicate `{:?}` added", data);
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
