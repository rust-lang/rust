// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use middle::infer::InferCtxt;
use middle::ty::{self, Ty, TypeFoldable};
use rustc_data_structures::obligation_forest::{Backtrace, ObligationForest, Error};
use std::iter;
use syntax::ast;
use util::common::ErrorReported;
use util::nodemap::{FnvHashMap, FnvHashSet, NodeMap};

use super::CodeAmbiguity;
use super::CodeProjectionError;
use super::CodeSelectionError;
use super::is_object_safe;
use super::FulfillmentError;
use super::FulfillmentErrorCode;
use super::ObligationCause;
use super::PredicateObligation;
use super::project;
use super::report_overflow_error_cycle;
use super::select::SelectionContext;
use super::Unimplemented;
use super::util::predicate_for_builtin_bound;

pub struct FulfilledPredicates<'tcx> {
    set: FnvHashSet<ty::Predicate<'tcx>>
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
    // a simple cache that aims to cache *exact duplicate obligations*
    // and avoid adding them twice. This serves a different purpose
    // than the `SelectionCache`: it avoids duplicate errors and
    // permits recursive obligations, which are often generated from
    // traits like `Send` et al.
    //
    // Note that because of type inference, a predicate can still
    // occur twice in the predicates list, for example when 2
    // initially-distinct type variables are unified after being
    // inserted. Deduplicating the predicate set on selection had a
    // significant performance cost the last time I checked.
    duplicate_set: FulfilledPredicates<'tcx>,

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
}

#[derive(Clone)]
pub struct RegionObligation<'tcx> {
    pub sub_region: ty::Region,
    pub sup_type: Ty<'tcx>,
    pub cause: ObligationCause<'tcx>,
}

#[derive(Clone, Debug)]
pub struct PendingPredicateObligation<'tcx> {
    pub obligation: PredicateObligation<'tcx>,
    pub stalled_on: Vec<Ty<'tcx>>,
}

impl<'tcx> FulfillmentContext<'tcx> {
    /// Creates a new fulfillment context.
    pub fn new() -> FulfillmentContext<'tcx> {
        FulfillmentContext {
            duplicate_set: FulfilledPredicates::new(),
            predicates: ObligationForest::new(),
            region_obligations: NodeMap(),
        }
    }

    /// "Normalize" a projection type `<SomeType as SomeTrait>::X` by
    /// creating a fresh type variable `$0` as well as a projection
    /// predicate `<SomeType as SomeTrait>::X == $0`. When the
    /// inference engine runs, it will attempt to find an impl of
    /// `SomeTrait` or a where clause that lets us unify `$0` with
    /// something concrete. If this fails, we'll unify `$0` with
    /// `projection_ty` again.
    pub fn normalize_projection_type<'a>(&mut self,
                                         infcx: &InferCtxt<'a,'tcx>,
                                         projection_ty: ty::ProjectionTy<'tcx>,
                                         cause: ObligationCause<'tcx>)
                                         -> Ty<'tcx>
    {
        debug!("normalize_associated_type(projection_ty={:?})",
               projection_ty);

        assert!(!projection_ty.has_escaping_regions());

        // FIXME(#20304) -- cache

        let mut selcx = SelectionContext::new(infcx);
        let normalized = project::normalize_projection_type(&mut selcx, projection_ty, cause, 0);

        for obligation in normalized.obligations {
            self.register_predicate_obligation(infcx, obligation);
        }

        debug!("normalize_associated_type: result={:?}", normalized.value);

        normalized.value
    }

    pub fn register_builtin_bound<'a>(&mut self,
                                      infcx: &InferCtxt<'a,'tcx>,
                                      ty: Ty<'tcx>,
                                      builtin_bound: ty::BuiltinBound,
                                      cause: ObligationCause<'tcx>)
    {
        match predicate_for_builtin_bound(infcx.tcx, cause, builtin_bound, 0, ty) {
            Ok(predicate) => {
                self.register_predicate_obligation(infcx, predicate);
            }
            Err(ErrorReported) => { }
        }
    }

    pub fn register_region_obligation<'a>(&mut self,
                                          t_a: Ty<'tcx>,
                                          r_b: ty::Region,
                                          cause: ObligationCause<'tcx>)
    {
        register_region_obligation(t_a, r_b, cause, &mut self.region_obligations);
    }

    pub fn register_predicate_obligation<'a>(&mut self,
                                             infcx: &InferCtxt<'a,'tcx>,
                                             obligation: PredicateObligation<'tcx>)
    {
        // this helps to reduce duplicate errors, as well as making
        // debug output much nicer to read and so on.
        let obligation = infcx.resolve_type_vars_if_possible(&obligation);

        assert!(!obligation.has_escaping_regions());

        if self.is_duplicate_or_add(infcx.tcx, &obligation.predicate) {
            debug!("register_predicate({:?}) -- already seen, skip", obligation);
            return;
        }

        debug!("register_predicate({:?})", obligation);
        let obligation = PendingPredicateObligation {
            obligation: obligation,
            stalled_on: vec![]
        };
        self.predicates.push_root(obligation);
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

    pub fn select_all_or_error<'a>(&mut self,
                                   infcx: &InferCtxt<'a,'tcx>)
                                   -> Result<(),Vec<FulfillmentError<'tcx>>>
    {
        try!(self.select_where_possible(infcx));
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

    pub fn select_where_possible<'a>(&mut self,
                                     infcx: &InferCtxt<'a,'tcx>)
                                     -> Result<(),Vec<FulfillmentError<'tcx>>>
    {
        let mut selcx = SelectionContext::new(infcx);
        self.select(&mut selcx)
    }

    pub fn pending_obligations(&self) -> Vec<PendingPredicateObligation<'tcx>> {
        self.predicates.pending_obligations()
    }

    fn is_duplicate_or_add(&mut self,
                           tcx: &ty::ctxt<'tcx>,
                           predicate: &ty::Predicate<'tcx>)
                           -> bool {
        // For "global" predicates -- that is, predicates that don't
        // involve type parameters, inference variables, or regions
        // other than 'static -- we can check the cache in the tcx,
        // which allows us to leverage work from other threads. Note
        // that we don't add anything to this cache yet (unlike the
        // local cache).  This is because the tcx cache maintains the
        // invariant that it only contains things that have been
        // proven, and we have not yet proven that `predicate` holds.
        if predicate.is_global() && tcx.fulfilled_predicates.borrow().is_duplicate(predicate) {
            return true;
        }

        // If `predicate` is not global, or not present in the tcx
        // cache, we can still check for it in our local cache and add
        // it if not present. Note that if we find this predicate in
        // the local cache we can stop immediately, without reporting
        // any errors, even though we don't know yet if it is
        // true. This is because, while we don't yet know if the
        // predicate holds, we know that this same fulfillment context
        // already is in the process of finding out.
        self.duplicate_set.is_duplicate_or_add(predicate)
    }

    /// Attempts to select obligations using `selcx`. If `only_new_obligations` is true, then it
    /// only attempts to select obligations that haven't been seen before.
    fn select<'a>(&mut self,
                  selcx: &mut SelectionContext<'a, 'tcx>)
                  -> Result<(),Vec<FulfillmentError<'tcx>>>
    {
        debug!("select(obligation-forest-size={})", self.predicates.len());

        let mut errors = Vec::new();

        loop {
            debug!("select_where_possible: starting another iteration");

            // Process pending obligations.
            let outcome = {
                let region_obligations = &mut self.region_obligations;
                self.predicates.process_obligations(
                    |obligation, backtrace| process_predicate(selcx,
                                                              obligation,
                                                              backtrace,
                                                              region_obligations))
            };

            debug!("select_where_possible: outcome={:?}", outcome);

            // these are obligations that were proven to be true.
            for pending_obligation in outcome.successful {
                let predicate = &pending_obligation.obligation.predicate;
                if predicate.is_global() {
                    selcx.tcx().fulfilled_predicates.borrow_mut()
                                                    .is_duplicate_or_add(predicate);
                }
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

/// Like `process_predicate1`, but wrap result into a pending predicate.
fn process_predicate<'a,'tcx>(selcx: &mut SelectionContext<'a,'tcx>,
                              pending_obligation: &mut PendingPredicateObligation<'tcx>,
                              backtrace: Backtrace<PendingPredicateObligation<'tcx>>,
                              region_obligations: &mut NodeMap<Vec<RegionObligation<'tcx>>>)
                              -> Result<Option<Vec<PendingPredicateObligation<'tcx>>>,
                                        FulfillmentErrorCode<'tcx>>
{
    match process_predicate1(selcx, pending_obligation, backtrace, region_obligations) {
        Ok(Some(v)) => {
            // FIXME the right thing to do here, I think, is to permit
            // DAGs. That is, we should detect whenever this predicate
            // has appeared somewhere in the current tree./ If it's a
            // parent, that's a cycle, and we should either error out
            // or consider it ok. But if it's NOT a parent, we can
            // ignore it, since it will be proven (or not) separately.
            // However, this is a touch tricky, so I'm doing something
            // a bit hackier for now so that the `huge-struct.rs` passes.

            let retain_vec: Vec<_> = {
                let mut dedup = FnvHashSet();
                v.iter()
                 .map(|o| {
                     // Screen out obligations that we know globally
                     // are true. This should really be the DAG check
                     // mentioned above.
                     if
                         o.predicate.is_global() &&
                         selcx.tcx().fulfilled_predicates.borrow().is_duplicate(&o.predicate)
                     {
                         return false;
                     }

                     // If we see two siblings that are exactly the
                     // same, no need to add them twice.
                     if !dedup.insert(&o.predicate) {
                         return false;
                     }

                     true
                 })
                 .collect()
            };

            let pending_predicate_obligations =
                v.into_iter()
                 .zip(retain_vec)
                 .flat_map(|(o, retain)| {
                     if retain {
                         Some(PendingPredicateObligation {
                             obligation: o,
                             stalled_on: vec![]
                         })
                     } else {
                         None
                     }
                 })
                .collect();

            Ok(Some(pending_predicate_obligations))
        }
        Ok(None) => Ok(None),
        Err(e) => Err(e)
    }
}

/// Processes a predicate obligation and returns either:
/// - `Ok(Some(v))` if the predicate is true, presuming that `v` are also true
/// - `Ok(None)` if we don't have enough info to be sure
/// - `Err` if the predicate does not hold
fn process_predicate1<'a,'tcx>(selcx: &mut SelectionContext<'a,'tcx>,
                               pending_obligation: &mut PendingPredicateObligation<'tcx>,
                               backtrace: Backtrace<PendingPredicateObligation<'tcx>>,
                               region_obligations: &mut NodeMap<Vec<RegionObligation<'tcx>>>)
                               -> Result<Option<Vec<PredicateObligation<'tcx>>>,
                                         FulfillmentErrorCode<'tcx>>
{
    // if we were stalled on some unresolved variables, first check
    // whether any of them have been resolved; if not, don't bother
    // doing more work yet
    if !pending_obligation.stalled_on.is_empty() {
        if pending_obligation.stalled_on.iter().all(|&ty| {
            let resolved_ty = selcx.infcx().resolve_type_vars_if_possible(&ty);
            resolved_ty == ty // nothing changed here
        }) {
            debug!("process_predicate: pending obligation {:?} still stalled on {:?}",
                   selcx.infcx().resolve_type_vars_if_possible(&pending_obligation.obligation),
                   pending_obligation.stalled_on);
            return Ok(None);
        }
        pending_obligation.stalled_on = vec![];
    }

    let obligation = &pending_obligation.obligation;

    // If we exceed the recursion limit, take a moment to look for a
    // cycle so we can give a better error report from here, where we
    // have more context.
    let recursion_limit = selcx.tcx().sess.recursion_limit.get();
    if obligation.recursion_depth >= recursion_limit {
        if let Some(cycle) = scan_for_cycle(obligation, &backtrace) {
            report_overflow_error_cycle(selcx.infcx(), &cycle);
        }
    }

    match obligation.predicate {
        ty::Predicate::Trait(ref data) => {
            if coinductive_match(selcx, obligation, data, &backtrace) {
                return Ok(Some(vec![]));
            }

            let trait_obligation = obligation.with(data.clone());
            match selcx.select(&trait_obligation) {
                Ok(Some(vtable)) => {
                    Ok(Some(vtable.nested_obligations()))
                }
                Ok(None) => {
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
                    pending_obligation.stalled_on =
                        data.skip_binder() // ok b/c this check doesn't care about regions
                        .input_types()
                        .iter()
                        .map(|t| selcx.infcx().resolve_type_vars_if_possible(t))
                        .filter(|t| t.has_infer_types())
                        .flat_map(|t| t.walk())
                        .filter(|t| match t.sty { ty::TyInfer(_) => true, _ => false })
                        .collect();

                    debug!("process_predicate: pending obligation {:?} now stalled on {:?}",
                           selcx.infcx().resolve_type_vars_if_possible(obligation),
                           pending_obligation.stalled_on);

                    Ok(None)
                }
                Err(selection_err) => {
                    Err(CodeSelectionError(selection_err))
                }
            }
        }

        ty::Predicate::Equate(ref binder) => {
            match selcx.infcx().equality_predicate(obligation.cause.span, binder) {
                Ok(()) => Ok(Some(Vec::new())),
                Err(_) => Err(CodeSelectionError(Unimplemented)),
            }
        }

        ty::Predicate::RegionOutlives(ref binder) => {
            match selcx.infcx().region_outlives_predicate(obligation.cause.span, binder) {
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
                            register_region_obligation(t_a, ty::ReStatic,
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
                Ok(v) => Ok(v),
                Err(e) => Err(CodeProjectionError(e))
            }
        }

        ty::Predicate::ObjectSafe(trait_def_id) => {
            if !is_object_safe(selcx.tcx(), trait_def_id) {
                Err(CodeSelectionError(Unimplemented))
            } else {
                Ok(Some(Vec::new()))
            }
        }

        ty::Predicate::WellFormed(ty) => {
            Ok(ty::wf::obligations(selcx.infcx(), obligation.cause.body_id,
                                   ty, obligation.cause.span))
        }
    }
}

/// For defaulted traits, we use a co-inductive strategy to solve, so
/// that recursion is ok. This routine returns true if the top of the
/// stack (`top_obligation` and `top_data`):
/// - is a defaulted trait, and
/// - it also appears in the backtrace at some position `X`; and,
/// - all the predicates at positions `X..` between `X` an the top are
///   also defaulted traits.
fn coinductive_match<'a,'tcx>(selcx: &mut SelectionContext<'a,'tcx>,
                              top_obligation: &PredicateObligation<'tcx>,
                              top_data: &ty::PolyTraitPredicate<'tcx>,
                              backtrace: &Backtrace<PendingPredicateObligation<'tcx>>)
                              -> bool
{
    if selcx.tcx().trait_has_default_impl(top_data.def_id()) {
        debug!("coinductive_match: top_data={:?}", top_data);
        for bt_obligation in backtrace.clone() {
            debug!("coinductive_match: bt_obligation={:?}", bt_obligation);

            // *Everything* in the backtrace must be a defaulted trait.
            match bt_obligation.obligation.predicate {
                ty::Predicate::Trait(ref data) => {
                    if !selcx.tcx().trait_has_default_impl(data.def_id()) {
                        debug!("coinductive_match: trait does not have default impl");
                        break;
                    }
                }
                _ => { break; }
            }

            // And we must find a recursive match.
            if bt_obligation.obligation.predicate == top_obligation.predicate {
                debug!("coinductive_match: found a match in the backtrace");
                return true;
            }
        }
    }

    false
}

fn scan_for_cycle<'a,'tcx>(top_obligation: &PredicateObligation<'tcx>,
                           backtrace: &Backtrace<PendingPredicateObligation<'tcx>>)
                           -> Option<Vec<PredicateObligation<'tcx>>>
{
    let mut map = FnvHashMap();
    let all_obligations =
        || iter::once(top_obligation)
               .chain(backtrace.clone()
                               .map(|p| &p.obligation));
    for (index, bt_obligation) in all_obligations().enumerate() {
        if let Some(&start) = map.get(&bt_obligation.predicate) {
            // Found a cycle starting at position `start` and running
            // until the current position (`index`).
            return Some(all_obligations().skip(start).take(index - start + 1).cloned().collect());
        } else {
            map.insert(bt_obligation.predicate.clone(), index);
        }
    }
    None
}

fn register_region_obligation<'tcx>(t_a: Ty<'tcx>,
                                    r_b: ty::Region,
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

impl<'tcx> FulfilledPredicates<'tcx> {
    pub fn new() -> FulfilledPredicates<'tcx> {
        FulfilledPredicates {
            set: FnvHashSet()
        }
    }

    pub fn is_duplicate(&self, key: &ty::Predicate<'tcx>) -> bool {
        self.set.contains(key)
    }

    fn is_duplicate_or_add(&mut self, key: &ty::Predicate<'tcx>) -> bool {
        !self.set.insert(key.clone())
    }
}

fn to_fulfillment_error<'tcx>(
    error: Error<PendingPredicateObligation<'tcx>, FulfillmentErrorCode<'tcx>>)
    -> FulfillmentError<'tcx>
{
    let obligation = error.backtrace.into_iter().next().unwrap().obligation;
    FulfillmentError::new(obligation, error.error)
}
