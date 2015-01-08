// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use middle::infer::{InferCtxt};
use middle::mem_categorization::Typer;
use middle::ty::{self, RegionEscape, Ty};
use std::collections::HashSet;
use std::collections::hash_map::Entry::{Occupied, Vacant};
use std::default::Default;
use syntax::ast;
use util::common::ErrorReported;
use util::ppaux::Repr;
use util::nodemap::NodeMap;

use super::CodeAmbiguity;
use super::CodeProjectionError;
use super::CodeSelectionError;
use super::FulfillmentError;
use super::ObligationCause;
use super::PredicateObligation;
use super::project;
use super::select::SelectionContext;
use super::Unimplemented;
use super::util::predicate_for_builtin_bound;

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
    duplicate_set: HashSet<ty::Predicate<'tcx>>,

    // A list of all obligations that have been registered with this
    // fulfillment context.
    predicates: Vec<PredicateObligation<'tcx>>,

    // Remembers the count of trait obligations that we have already
    // attempted to select. This is used to avoid repeating work
    // when `select_new_obligations` is called.
    attempted_mark: uint,

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

pub struct RegionObligation<'tcx> {
    pub sub_region: ty::Region,
    pub sup_type: Ty<'tcx>,
    pub cause: ObligationCause<'tcx>,
}

impl<'tcx> FulfillmentContext<'tcx> {
    pub fn new() -> FulfillmentContext<'tcx> {
        FulfillmentContext {
            duplicate_set: HashSet::new(),
            predicates: Vec::new(),
            attempted_mark: 0,
            region_obligations: NodeMap::new(),
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
                                         typer: &ty::UnboxedClosureTyper<'tcx>,
                                         projection_ty: ty::ProjectionTy<'tcx>,
                                         cause: ObligationCause<'tcx>)
                                         -> Ty<'tcx>
    {
        debug!("normalize_associated_type(projection_ty={})",
               projection_ty.repr(infcx.tcx));

        assert!(!projection_ty.has_escaping_regions());

        // FIXME(#20304) -- cache

        let mut selcx = SelectionContext::new(infcx, typer);
        let normalized = project::normalize_projection_type(&mut selcx, projection_ty, cause, 0);

        for obligation in normalized.obligations.into_iter() {
            self.register_predicate_obligation(infcx, obligation);
        }

        debug!("normalize_associated_type: result={}", normalized.value.repr(infcx.tcx));

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
                                          infcx: &InferCtxt<'a,'tcx>,
                                          t_a: Ty<'tcx>,
                                          r_b: ty::Region,
                                          cause: ObligationCause<'tcx>)
    {
        register_region_obligation(infcx.tcx, t_a, r_b, cause, &mut self.region_obligations);
    }

    pub fn register_predicate_obligation<'a>(&mut self,
                                             infcx: &InferCtxt<'a,'tcx>,
                                             obligation: PredicateObligation<'tcx>)
    {
        // this helps to reduce duplicate errors, as well as making
        // debug output much nicer to read and so on.
        let obligation = infcx.resolve_type_vars_if_possible(&obligation);

        if !self.duplicate_set.insert(obligation.predicate.clone()) {
            debug!("register_predicate({}) -- already seen, skip", obligation.repr(infcx.tcx));
            return;
        }

        debug!("register_predicate({})", obligation.repr(infcx.tcx));
        self.predicates.push(obligation);
    }

    pub fn region_obligations(&self,
                              body_id: ast::NodeId)
                              -> &[RegionObligation<'tcx>]
    {
        match self.region_obligations.get(&body_id) {
            None => Default::default(),
            Some(vec) => vec.as_slice(),
        }
    }

    pub fn select_all_or_error<'a>(&mut self,
                                   infcx: &InferCtxt<'a,'tcx>,
                                   typer: &ty::UnboxedClosureTyper<'tcx>)
                                   -> Result<(),Vec<FulfillmentError<'tcx>>>
    {
        try!(self.select_where_possible(infcx, typer));

        // Anything left is ambiguous.
        let errors: Vec<FulfillmentError> =
            self.predicates
            .iter()
            .map(|o| FulfillmentError::new((*o).clone(), CodeAmbiguity))
            .collect();

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// Attempts to select obligations that were registered since the call to a selection routine.
    /// This is used by the type checker to eagerly attempt to resolve obligations in hopes of
    /// gaining type information. It'd be equally valid to use `select_where_possible` but it
    /// results in `O(n^2)` performance (#18208).
    pub fn select_new_obligations<'a>(&mut self,
                                      infcx: &InferCtxt<'a,'tcx>,
                                      typer: &ty::UnboxedClosureTyper<'tcx>)
                                      -> Result<(),Vec<FulfillmentError<'tcx>>>
    {
        let mut selcx = SelectionContext::new(infcx, typer);
        self.select(&mut selcx, true)
    }

    pub fn select_where_possible<'a>(&mut self,
                                     infcx: &InferCtxt<'a,'tcx>,
                                     typer: &ty::UnboxedClosureTyper<'tcx>)
                                     -> Result<(),Vec<FulfillmentError<'tcx>>>
    {
        let mut selcx = SelectionContext::new(infcx, typer);
        self.select(&mut selcx, false)
    }

    pub fn pending_obligations(&self) -> &[PredicateObligation<'tcx>] {
        &self.predicates[]
    }

    /// Attempts to select obligations using `selcx`. If `only_new_obligations` is true, then it
    /// only attempts to select obligations that haven't been seen before.
    fn select<'a>(&mut self,
                  selcx: &mut SelectionContext<'a, 'tcx>,
                  only_new_obligations: bool)
                  -> Result<(),Vec<FulfillmentError<'tcx>>>
    {
        debug!("select({} obligations, only_new_obligations={}) start",
               self.predicates.len(),
               only_new_obligations);

        let mut errors = Vec::new();

        loop {
            let count = self.predicates.len();

            debug!("select_where_possible({} obligations) iteration",
                   count);

            let mut new_obligations = Vec::new();

            // If we are only attempting obligations we haven't seen yet,
            // then set `skip` to the number of obligations we've already
            // seen.
            let mut skip = if only_new_obligations {
                self.attempted_mark
            } else {
                0
            };

            // First pass: walk each obligation, retaining
            // only those that we cannot yet process.
            {
                let region_obligations = &mut self.region_obligations;
                self.predicates.retain(|predicate| {
                    // Hack: Retain does not pass in the index, but we want
                    // to avoid processing the first `start_count` entries.
                    let processed =
                        if skip == 0 {
                            process_predicate(selcx, predicate,
                                              &mut new_obligations, &mut errors, region_obligations)
                        } else {
                            skip -= 1;
                            false
                        };
                    !processed
                });
            }

            self.attempted_mark = self.predicates.len();

            if self.predicates.len() == count {
                // Nothing changed.
                break;
            }

            // Now go through all the successful ones,
            // registering any nested obligations for the future.
            for new_obligation in new_obligations.into_iter() {
                self.register_predicate_obligation(selcx.infcx(), new_obligation);
            }
        }

        debug!("select({} obligations, {} errors) done",
               self.predicates.len(),
               errors.len());

        if errors.len() == 0 {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

fn process_predicate<'a,'tcx>(selcx: &mut SelectionContext<'a,'tcx>,
                              obligation: &PredicateObligation<'tcx>,
                              new_obligations: &mut Vec<PredicateObligation<'tcx>>,
                              errors: &mut Vec<FulfillmentError<'tcx>>,
                              region_obligations: &mut NodeMap<Vec<RegionObligation<'tcx>>>)
                              -> bool
{
    /*!
     * Processes a predicate obligation and modifies the appropriate
     * output array with the successful/error result.  Returns `false`
     * if the predicate could not be processed due to insufficient
     * type inference.
     */

    let tcx = selcx.tcx();
    match obligation.predicate {
        ty::Predicate::Trait(ref data) => {
            let trait_obligation = obligation.with(data.clone());
            match selcx.select(&trait_obligation) {
                Ok(None) => {
                    false
                }
                Ok(Some(s)) => {
                    s.map_move_nested(|p| new_obligations.push(p));
                    true
                }
                Err(selection_err) => {
                    debug!("predicate: {} error: {}",
                           obligation.repr(tcx),
                           selection_err.repr(tcx));
                    errors.push(
                        FulfillmentError::new(
                            obligation.clone(),
                            CodeSelectionError(selection_err)));
                    true
                }
            }
        }

        ty::Predicate::Equate(ref binder) => {
            match selcx.infcx().equality_predicate(obligation.cause.span, binder) {
                Ok(()) => { }
                Err(_) => {
                    errors.push(
                        FulfillmentError::new(
                            obligation.clone(),
                            CodeSelectionError(Unimplemented)));
                }
            }
            true
        }

        ty::Predicate::RegionOutlives(ref binder) => {
            match selcx.infcx().region_outlives_predicate(obligation.cause.span, binder) {
                Ok(()) => { }
                Err(_) => {
                    errors.push(
                        FulfillmentError::new(
                            obligation.clone(),
                            CodeSelectionError(Unimplemented)));
                }
            }

            true
        }

        ty::Predicate::TypeOutlives(ref binder) => {
            // For now, we just check that there are no higher-ranked
            // regions.  If there are, we will call this obligation an
            // error. Eventually we should be able to support some
            // cases here, I imagine (e.g., `for<'a> int : 'a`).
            if ty::count_late_bound_regions(selcx.tcx(), binder) != 0 {
                errors.push(
                    FulfillmentError::new(
                        obligation.clone(),
                        CodeSelectionError(Unimplemented)));
            } else {
                let ty::OutlivesPredicate(t_a, r_b) = binder.0;
                register_region_obligation(tcx, t_a, r_b,
                                           obligation.cause.clone(),
                                           region_obligations);
            }
            true
        }

        ty::Predicate::Projection(ref data) => {
            let project_obligation = obligation.with(data.clone());
            let result = project::poly_project_and_unify_type(selcx, &project_obligation);
            debug!("poly_project_and_unify_type({}) = {}",
                   project_obligation.repr(tcx),
                   result.repr(tcx));
            match result {
                Ok(Some(obligations)) => {
                    new_obligations.extend(obligations.into_iter());
                    true
                }
                Ok(None) => {
                    false
                }
                Err(err) => {
                    errors.push(
                        FulfillmentError::new(
                            obligation.clone(),
                            CodeProjectionError(err)));
                    true
                }
            }
        }
    }
}

impl<'tcx> Repr<'tcx> for RegionObligation<'tcx> {
    fn repr(&self, tcx: &ty::ctxt<'tcx>) -> String {
        format!("RegionObligation(sub_region={}, sup_type={})",
                self.sub_region.repr(tcx),
                self.sup_type.repr(tcx))
    }
}

fn register_region_obligation<'tcx>(tcx: &ty::ctxt<'tcx>,
                                    t_a: Ty<'tcx>,
                                    r_b: ty::Region,
                                    cause: ObligationCause<'tcx>,
                                    region_obligations: &mut NodeMap<Vec<RegionObligation<'tcx>>>)
{
    let region_obligation = RegionObligation { sup_type: t_a,
                                               sub_region: r_b,
                                               cause: cause };

    debug!("register_region_obligation({})",
           region_obligation.repr(tcx));

    match region_obligations.entry(region_obligation.cause.body_id) {
        Vacant(entry) => { entry.insert(vec![region_obligation]); },
        Occupied(mut entry) => { entry.get_mut().push(region_obligation); },
    }

}
