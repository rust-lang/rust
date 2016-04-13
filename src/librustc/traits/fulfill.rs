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
use ty::{self, Ty, TyCtxt, TypeFoldable, ToPolyTraitRef};
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

pub struct GlobalFulfilledPredicates<'tcx> {
    set: FnvHashSet<ty::PolyTraitPredicate<'tcx>>,
    dep_graph: DepGraph,
}

#[derive(Debug)]
pub struct LocalFulfilledPredicates<'tcx> {
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
    duplicate_set: LocalFulfilledPredicates<'tcx>,

    // A list of all obligations that have been registered with this
    // fulfillment context.
    predicates: ObligationForest<PendingPredicateObligation<'tcx>,
                                 LocalFulfilledPredicates<'tcx>>,

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
            duplicate_set: LocalFulfilledPredicates::new(),
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
            debug!("register_predicate_obligation({:?}) -- already seen, skip", obligation);
            return;
        }

        debug!("register_predicate_obligation({:?})", obligation);
        let obligation = PendingPredicateObligation {
            obligation: obligation,
            stalled_on: vec![]
        };
        self.predicates.push_tree(obligation, LocalFulfilledPredicates::new());
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
                           tcx: &TyCtxt<'tcx>,
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
        if tcx.fulfilled_predicates.borrow().check_duplicate(predicate) {
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
            debug!("select: starting another iteration");

            // Process pending obligations.
            let outcome = {
                let region_obligations = &mut self.region_obligations;
                self.predicates.process_obligations(
                    |obligation, tree, backtrace| process_predicate(selcx,
                                                                     tree,
                                                                     obligation,
                                                                     backtrace,
                                                                     region_obligations))
            };

            debug!("select: outcome={:?}", outcome);

            // these are obligations that were proven to be true.
            for pending_obligation in outcome.completed {
                let predicate = &pending_obligation.obligation.predicate;
                selcx.tcx().fulfilled_predicates.borrow_mut().add_if_global(predicate);
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
                              tree_cache: &mut LocalFulfilledPredicates<'tcx>,
                              pending_obligation: &mut PendingPredicateObligation<'tcx>,
                              backtrace: Backtrace<PendingPredicateObligation<'tcx>>,
                              region_obligations: &mut NodeMap<Vec<RegionObligation<'tcx>>>)
                              -> Result<Option<Vec<PendingPredicateObligation<'tcx>>>,
                                        FulfillmentErrorCode<'tcx>>
{
    match process_predicate1(selcx, pending_obligation, region_obligations) {
        Ok(Some(v)) => process_child_obligations(selcx,
                                                 tree_cache,
                                                 &pending_obligation.obligation,
                                                 backtrace,
                                                 v),
        Ok(None) => Ok(None),
        Err(e) => Err(e)
    }
}

fn process_child_obligations<'a,'tcx>(
    selcx: &mut SelectionContext<'a,'tcx>,
    tree_cache: &mut LocalFulfilledPredicates<'tcx>,
    pending_obligation: &PredicateObligation<'tcx>,
    backtrace: Backtrace<PendingPredicateObligation<'tcx>>,
    child_obligations: Vec<PredicateObligation<'tcx>>)
    -> Result<Option<Vec<PendingPredicateObligation<'tcx>>>,
              FulfillmentErrorCode<'tcx>>
{
    // FIXME(#30977) The code below is designed to detect (and
    // permit) DAGs, while still ensuring that the reasoning
    // is acyclic. However, it does a few things
    // suboptimally. For example, it refreshes type variables
    // a lot, probably more than needed, but also less than
    // you might want.
    //
    //   - more than needed: I want to be very sure we don't
    //     accidentally treat a cycle as a DAG, so I am
    //     refreshing type variables as we walk the ancestors;
    //     but we are going to repeat this a lot, which is
    //     sort of silly, and it would be nicer to refresh
    //     them *in place* so that later predicate processing
    //     can benefit from the same work;
    //   - less than you might want: we only add items in the cache here,
    //     but maybe we learn more about type variables and could add them into
    //     the cache later on.

    let tcx = selcx.tcx();

    let mut ancestor_set = AncestorSet::new(&backtrace);

    let pending_predicate_obligations: Vec<_> =
        child_obligations
        .into_iter()
        .filter_map(|obligation| {
            // Probably silly, but remove any inference
            // variables. This is actually crucial to the ancestor
            // check marked (*) below, but it's not clear that it
            // makes sense to ALWAYS do it.
            let obligation = selcx.infcx().resolve_type_vars_if_possible(&obligation);

            // Screen out obligations that we know globally
            // are true.
            if tcx.fulfilled_predicates.borrow().check_duplicate(&obligation.predicate) {
                return None;
            }

            // Check whether this obligation appears
            // somewhere else in the tree. If not, we have to
            // process it for sure.
            if !tree_cache.is_duplicate_or_add(&obligation.predicate) {
                return Some(PendingPredicateObligation {
                    obligation: obligation,
                    stalled_on: vec![]
                });
            }

            debug!("process_child_obligations: duplicate={:?}",
                   obligation.predicate);

            // OK, the obligation appears elsewhere in the tree.
            // This is either a fatal error or else something we can
            // ignore. If the obligation appears in our *ancestors*
            // (rather than some more distant relative), that
            // indicates a cycle. Cycles are either considered
            // resolved (if this is a coinductive case) or a fatal
            // error.
            if let Some(index) = ancestor_set.has(selcx.infcx(), &obligation.predicate) {
                //                            ~~~ (*) see above
                debug!("process_child_obligations: cycle index = {}", index);

                let backtrace = backtrace.clone();
                let cycle: Vec<_> =
                    iter::once(&obligation)
                    .chain(Some(pending_obligation))
                    .chain(backtrace.take(index + 1).map(|p| &p.obligation))
                    .cloned()
                    .collect();
                if coinductive_match(selcx, &cycle) {
                    debug!("process_child_obligations: coinductive match");
                    None
                } else {
                    report_overflow_error_cycle(selcx.infcx(), &cycle);
                }
            } else {
                // Not a cycle. Just ignore this obligation then,
                // we're already in the process of proving it.
                debug!("process_child_obligations: not a cycle");
                None
            }
        })
        .collect();

    Ok(Some(pending_predicate_obligations))
}

struct AncestorSet<'b, 'tcx: 'b> {
    populated: bool,
    cache: FnvHashMap<ty::Predicate<'tcx>, usize>,
    backtrace: Backtrace<'b, PendingPredicateObligation<'tcx>>,
}

impl<'b, 'tcx> AncestorSet<'b, 'tcx> {
    fn new(backtrace: &Backtrace<'b, PendingPredicateObligation<'tcx>>) -> Self {
        AncestorSet {
            populated: false,
            cache: FnvHashMap(),
            backtrace: backtrace.clone(),
        }
    }

    /// Checks whether any of the ancestors in the backtrace are equal
    /// to `predicate` (`predicate` is assumed to be fully
    /// type-resolved).  Returns `None` if not; otherwise, returns
    /// `Some` with the index within the backtrace.
    fn has<'a>(&mut self,
               infcx: &InferCtxt<'a, 'tcx>,
               predicate: &ty::Predicate<'tcx>)
               -> Option<usize> {
        // the first time, we have to populate the cache
        if !self.populated {
            let backtrace = self.backtrace.clone();
            for (index, ancestor) in backtrace.enumerate() {
                // Ugh. This just feels ridiculously
                // inefficient.  But we need to compare
                // predicates without being concerned about
                // the vagaries of type inference, so for now
                // just ensure that they are always
                // up-to-date. (I suppose we could just use a
                // snapshot and check if they are unifiable?)
                let resolved_predicate =
                    infcx.resolve_type_vars_if_possible(
                        &ancestor.obligation.predicate);

                // Though we try to avoid it, it can happen that a
                // cycle already exists in the predecessors. This
                // happens if the type variables were not fully known
                // at the time that the ancestors were pushed. We'll
                // just ignore such cycles for now, on the premise
                // that they will repeat themselves and we'll deal
                // with them properly then.
                self.cache.entry(resolved_predicate)
                          .or_insert(index);
            }
            self.populated = true;
        }

        self.cache.get(predicate).cloned()
    }
}

/// Return the set of type variables contained in a trait ref
fn trait_ref_type_vars<'a, 'tcx>(selcx: &mut SelectionContext<'a, 'tcx>,
                                 t: ty::PolyTraitRef<'tcx>) -> Vec<Ty<'tcx>>
{
    t.skip_binder() // ok b/c this check doesn't care about regions
     .input_types()
     .iter()
     .map(|t| selcx.infcx().resolve_type_vars_if_possible(t))
     .filter(|t| t.has_infer_types())
     .flat_map(|t| t.walk())
     .filter(|t| match t.sty { ty::TyInfer(_) => true, _ => false })
     .collect()
}

/// Processes a predicate obligation and returns either:
/// - `Ok(Some(v))` if the predicate is true, presuming that `v` are also true
/// - `Ok(None)` if we don't have enough info to be sure
/// - `Err` if the predicate does not hold
fn process_predicate1<'a,'tcx>(selcx: &mut SelectionContext<'a,'tcx>,
                               pending_obligation: &mut PendingPredicateObligation<'tcx>,
                               region_obligations: &mut NodeMap<Vec<RegionObligation<'tcx>>>)
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
                    info!("selecting trait `{:?}` at depth {} yielded Ok(Some)",
                          data, obligation.recursion_depth);
                    Ok(Some(vtable.nested_obligations()))
                }
                Ok(None) => {
                    info!("selecting trait `{:?}` at depth {} yielded Ok(None)",
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
            match selcx.infcx().equality_predicate(obligation.cause.span, binder) {
                Ok(InferOk { obligations, .. }) => {
                    // FIXME(#32730) propagate obligations
                    assert!(obligations.is_empty());
                    Ok(Some(Vec::new()))
                },
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
            if !is_object_safe(selcx.tcx(), trait_def_id) {
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
fn coinductive_match<'a,'tcx>(selcx: &mut SelectionContext<'a,'tcx>,
                              cycle: &[PredicateObligation<'tcx>])
                              -> bool
{
    let len = cycle.len();

    assert_eq!(cycle[0].predicate, cycle[len - 1].predicate);

    cycle[0..len-1]
        .iter()
        .all(|bt_obligation| {
            let result = coinductive_obligation(selcx, bt_obligation);
            debug!("coinductive_match: bt_obligation={:?} coinductive={}",
                   bt_obligation, result);
            result
        })
}

fn coinductive_obligation<'a, 'tcx>(selcx: &SelectionContext<'a, 'tcx>,
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

impl<'tcx> LocalFulfilledPredicates<'tcx> {
    pub fn new() -> LocalFulfilledPredicates<'tcx> {
        LocalFulfilledPredicates {
            set: FnvHashSet()
        }
    }

    fn is_duplicate_or_add(&mut self, key: &ty::Predicate<'tcx>) -> bool {
        // For a `LocalFulfilledPredicates`, if we find a match, we
        // don't need to add a read edge to the dep-graph. This is
        // because it means that the predicate has already been
        // considered by this `FulfillmentContext`, and hence the
        // containing task will already have an edge. (Here we are
        // assuming each `FulfillmentContext` only gets used from one
        // task; but to do otherwise makes no sense)
        !self.set.insert(key.clone())
    }
}

impl<'tcx> GlobalFulfilledPredicates<'tcx> {
    pub fn new(dep_graph: DepGraph) -> GlobalFulfilledPredicates<'tcx> {
        GlobalFulfilledPredicates {
            set: FnvHashSet(),
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

            info!("check_duplicate_trait hit: `{:?}`", data);

            true
        } else {
            false
        }
    }

    fn add_if_global(&mut self, key: &ty::Predicate<'tcx>) {
        if let ty::Predicate::Trait(ref data) = *key {
            // We only add things to the global predicate registry
            // after the current task has proved them, and hence
            // already has the required read edges, so we don't need
            // to add any more edges here.
            if data.is_global() {
                if self.set.insert(data.clone()) {
                    debug!("add_if_global: global predicate `{:?}` added", data);
                    info!("check_duplicate_trait entry: `{:?}`", data);
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
