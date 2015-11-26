// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Code for projecting associated types out of trait references.

use super::elaborate_predicates;
use super::report_overflow_error;
use super::Obligation;
use super::ObligationCause;
use super::PredicateObligation;
use super::SelectionContext;
use super::SelectionError;
use super::VtableClosureData;
use super::VtableImplData;
use super::util;

use middle::infer::{self, TypeOrigin};
use middle::subst::Subst;
use middle::ty::{self, ToPredicate, RegionEscape, HasTypeFlags, ToPolyTraitRef, Ty};
use middle::ty::fold::{TypeFoldable, TypeFolder};
use syntax::parse::token;
use util::common::FN_OUTPUT_NAME;

pub type PolyProjectionObligation<'tcx> =
    Obligation<'tcx, ty::PolyProjectionPredicate<'tcx>>;

pub type ProjectionObligation<'tcx> =
    Obligation<'tcx, ty::ProjectionPredicate<'tcx>>;

pub type ProjectionTyObligation<'tcx> =
    Obligation<'tcx, ty::ProjectionTy<'tcx>>;

/// When attempting to resolve `<T as TraitRef>::Name` ...
#[derive(Debug)]
pub enum ProjectionTyError<'tcx> {
    /// ...we found multiple sources of information and couldn't resolve the ambiguity.
    TooManyCandidates,

    /// ...an error occurred matching `T : TraitRef`
    TraitSelectionError(SelectionError<'tcx>),
}

#[derive(Clone)]
pub struct MismatchedProjectionTypes<'tcx> {
    pub err: ty::error::TypeError<'tcx>
}

#[derive(PartialEq, Eq, Debug)]
enum ProjectionTyCandidate<'tcx> {
    // from a where-clause in the env or object type
    ParamEnv(ty::PolyProjectionPredicate<'tcx>),

    // from the definition of `Trait` when you have something like <<A as Trait>::B as Trait2>::C
    TraitDef(ty::PolyProjectionPredicate<'tcx>),

    // defined in an impl
    Impl(VtableImplData<'tcx, PredicateObligation<'tcx>>),

    // closure return type
    Closure(VtableClosureData<'tcx, PredicateObligation<'tcx>>),

    // fn pointer return type
    FnPointer(Ty<'tcx>),
}

struct ProjectionTyCandidateSet<'tcx> {
    vec: Vec<ProjectionTyCandidate<'tcx>>,
    ambiguous: bool
}

/// Evaluates constraints of the form:
///
///     for<...> <T as Trait>::U == V
///
/// If successful, this may result in additional obligations.
pub fn poly_project_and_unify_type<'cx,'tcx>(
    selcx: &mut SelectionContext<'cx,'tcx>,
    obligation: &PolyProjectionObligation<'tcx>)
    -> Result<Option<Vec<PredicateObligation<'tcx>>>, MismatchedProjectionTypes<'tcx>>
{
    debug!("poly_project_and_unify_type(obligation={:?})",
           obligation);

    let infcx = selcx.infcx();
    infcx.commit_if_ok(|snapshot| {
        let (skol_predicate, skol_map) =
            infcx.skolemize_late_bound_regions(&obligation.predicate, snapshot);

        let skol_obligation = obligation.with(skol_predicate);
        match project_and_unify_type(selcx, &skol_obligation) {
            Ok(result) => {
                match infcx.leak_check(&skol_map, snapshot) {
                    Ok(()) => Ok(infcx.plug_leaks(skol_map, snapshot, &result)),
                    Err(e) => Err(MismatchedProjectionTypes { err: e }),
                }
            }
            Err(e) => {
                Err(e)
            }
        }
    })
}

/// Evaluates constraints of the form:
///
///     <T as Trait>::U == V
///
/// If successful, this may result in additional obligations.
fn project_and_unify_type<'cx,'tcx>(
    selcx: &mut SelectionContext<'cx,'tcx>,
    obligation: &ProjectionObligation<'tcx>)
    -> Result<Option<Vec<PredicateObligation<'tcx>>>, MismatchedProjectionTypes<'tcx>>
{
    debug!("project_and_unify_type(obligation={:?})",
           obligation);

    let Normalized { value: normalized_ty, obligations } =
        match opt_normalize_projection_type(selcx,
                                            obligation.predicate.projection_ty.clone(),
                                            obligation.cause.clone(),
                                            obligation.recursion_depth) {
            Some(n) => n,
            None => {
                consider_unification_despite_ambiguity(selcx, obligation);
                return Ok(None);
            }
        };

    debug!("project_and_unify_type: normalized_ty={:?} obligations={:?}",
           normalized_ty,
           obligations);

    let infcx = selcx.infcx();
    let origin = TypeOrigin::RelateOutputImplTypes(obligation.cause.span);
    match infer::mk_eqty(infcx, true, origin, normalized_ty, obligation.predicate.ty) {
        Ok(()) => Ok(Some(obligations)),
        Err(err) => Err(MismatchedProjectionTypes { err: err }),
    }
}

fn consider_unification_despite_ambiguity<'cx,'tcx>(selcx: &mut SelectionContext<'cx,'tcx>,
                                                    obligation: &ProjectionObligation<'tcx>) {
    debug!("consider_unification_despite_ambiguity(obligation={:?})",
           obligation);

    let def_id = obligation.predicate.projection_ty.trait_ref.def_id;
    match selcx.tcx().lang_items.fn_trait_kind(def_id) {
        Some(_) => { }
        None => { return; }
    }

    let infcx = selcx.infcx();
    let self_ty = obligation.predicate.projection_ty.trait_ref.self_ty();
    let self_ty = infcx.shallow_resolve(self_ty);
    debug!("consider_unification_despite_ambiguity: self_ty.sty={:?}",
           self_ty.sty);
    match self_ty.sty {
        ty::TyClosure(closure_def_id, ref substs) => {
            let closure_typer = selcx.closure_typer();
            let closure_type = closure_typer.closure_type(closure_def_id, substs);
            let ty::Binder((_, ret_type)) =
                util::closure_trait_ref_and_return_type(infcx.tcx,
                                                        def_id,
                                                        self_ty,
                                                        &closure_type.sig,
                                                        util::TupleArgumentsFlag::No);
            // We don't have to normalize the return type here - this is only
            // reached for TyClosure: Fn inputs where the closure kind is
            // still unknown, which should only occur in typeck where the
            // closure type is already normalized.
            let (ret_type, _) =
                infcx.replace_late_bound_regions_with_fresh_var(
                    obligation.cause.span,
                    infer::AssocTypeProjection(obligation.predicate.projection_ty.item_name),
                    &ty::Binder(ret_type));

            debug!("consider_unification_despite_ambiguity: ret_type={:?}",
                   ret_type);
            let origin = TypeOrigin::RelateOutputImplTypes(obligation.cause.span);
            let obligation_ty = obligation.predicate.ty;
            match infer::mk_eqty(infcx, true, origin, obligation_ty, ret_type) {
                Ok(()) => { }
                Err(_) => { /* ignore errors */ }
            }
        }
        _ => { }
    }
}

/// Normalizes any associated type projections in `value`, replacing
/// them with a fully resolved type where possible. The return value
/// combines the normalized result and any additional obligations that
/// were incurred as result.
pub fn normalize<'a,'b,'tcx,T>(selcx: &'a mut SelectionContext<'b,'tcx>,
                               cause: ObligationCause<'tcx>,
                               value: &T)
                               -> Normalized<'tcx, T>
    where T : TypeFoldable<'tcx> + HasTypeFlags
{
    normalize_with_depth(selcx, cause, 0, value)
}

/// As `normalize`, but with a custom depth.
pub fn normalize_with_depth<'a,'b,'tcx,T>(selcx: &'a mut SelectionContext<'b,'tcx>,
                                          cause: ObligationCause<'tcx>,
                                          depth: usize,
                                          value: &T)
                                          -> Normalized<'tcx, T>
    where T : TypeFoldable<'tcx> + HasTypeFlags
{
    let mut normalizer = AssociatedTypeNormalizer::new(selcx, cause, depth);
    let result = normalizer.fold(value);

    Normalized {
        value: result,
        obligations: normalizer.obligations,
    }
}

struct AssociatedTypeNormalizer<'a,'b:'a,'tcx:'b> {
    selcx: &'a mut SelectionContext<'b,'tcx>,
    cause: ObligationCause<'tcx>,
    obligations: Vec<PredicateObligation<'tcx>>,
    depth: usize,
}

impl<'a,'b,'tcx> AssociatedTypeNormalizer<'a,'b,'tcx> {
    fn new(selcx: &'a mut SelectionContext<'b,'tcx>,
           cause: ObligationCause<'tcx>,
           depth: usize)
           -> AssociatedTypeNormalizer<'a,'b,'tcx>
    {
        AssociatedTypeNormalizer {
            selcx: selcx,
            cause: cause,
            obligations: vec!(),
            depth: depth,
        }
    }

    fn fold<T:TypeFoldable<'tcx> + HasTypeFlags>(&mut self, value: &T) -> T {
        let value = self.selcx.infcx().resolve_type_vars_if_possible(value);

        if !value.has_projection_types() {
            value.clone()
        } else {
            value.fold_with(self)
        }
    }
}

impl<'a,'b,'tcx> TypeFolder<'tcx> for AssociatedTypeNormalizer<'a,'b,'tcx> {
    fn tcx(&self) -> &ty::ctxt<'tcx> {
        self.selcx.tcx()
    }

    fn fold_ty(&mut self, ty: Ty<'tcx>) -> Ty<'tcx> {
        // We don't want to normalize associated types that occur inside of region
        // binders, because they may contain bound regions, and we can't cope with that.
        //
        // Example:
        //
        //     for<'a> fn(<T as Foo<&'a>>::A)
        //
        // Instead of normalizing `<T as Foo<&'a>>::A` here, we'll
        // normalize it when we instantiate those bound regions (which
        // should occur eventually).

        let ty = ty::fold::super_fold_ty(self, ty);
        match ty.sty {
            ty::TyProjection(ref data) if !data.has_escaping_regions() => { // (*)

                // (*) This is kind of hacky -- we need to be able to
                // handle normalization within binders because
                // otherwise we wind up a need to normalize when doing
                // trait matching (since you can have a trait
                // obligation like `for<'a> T::B : Fn(&'a int)`), but
                // we can't normalize with bound regions in scope. So
                // far now we just ignore binders but only normalize
                // if all bound regions are gone (and then we still
                // have to renormalize whenever we instantiate a
                // binder). It would be better to normalize in a
                // binding-aware fashion.

                let Normalized { value: ty, obligations } =
                    normalize_projection_type(self.selcx,
                                              data.clone(),
                                              self.cause.clone(),
                                              self.depth);
                self.obligations.extend(obligations);
                ty
            }

            _ => {
                ty
            }
        }
    }
}

#[derive(Clone)]
pub struct Normalized<'tcx,T> {
    pub value: T,
    pub obligations: Vec<PredicateObligation<'tcx>>,
}

pub type NormalizedTy<'tcx> = Normalized<'tcx, Ty<'tcx>>;

impl<'tcx,T> Normalized<'tcx,T> {
    pub fn with<U>(self, value: U) -> Normalized<'tcx,U> {
        Normalized { value: value, obligations: self.obligations }
    }
}

/// The guts of `normalize`: normalize a specific projection like `<T
/// as Trait>::Item`. The result is always a type (and possibly
/// additional obligations). If ambiguity arises, which implies that
/// there are unresolved type variables in the projection, we will
/// substitute a fresh type variable `$X` and generate a new
/// obligation `<T as Trait>::Item == $X` for later.
pub fn normalize_projection_type<'a,'b,'tcx>(
    selcx: &'a mut SelectionContext<'b,'tcx>,
    projection_ty: ty::ProjectionTy<'tcx>,
    cause: ObligationCause<'tcx>,
    depth: usize)
    -> NormalizedTy<'tcx>
{
    opt_normalize_projection_type(selcx, projection_ty.clone(), cause.clone(), depth)
        .unwrap_or_else(move || {
            // if we bottom out in ambiguity, create a type variable
            // and a deferred predicate to resolve this when more type
            // information is available.

            let ty_var = selcx.infcx().next_ty_var();
            let projection = ty::Binder(ty::ProjectionPredicate {
                projection_ty: projection_ty,
                ty: ty_var
            });
            let obligation = Obligation::with_depth(
                cause, depth + 1, projection.to_predicate());
            Normalized {
                value: ty_var,
                obligations: vec!(obligation)
            }
        })
}

/// The guts of `normalize`: normalize a specific projection like `<T
/// as Trait>::Item`. The result is always a type (and possibly
/// additional obligations). Returns `None` in the case of ambiguity,
/// which indicates that there are unbound type variables.
fn opt_normalize_projection_type<'a,'b,'tcx>(
    selcx: &'a mut SelectionContext<'b,'tcx>,
    projection_ty: ty::ProjectionTy<'tcx>,
    cause: ObligationCause<'tcx>,
    depth: usize)
    -> Option<NormalizedTy<'tcx>>
{
    debug!("normalize_projection_type(\
           projection_ty={:?}, \
           depth={})",
           projection_ty,
           depth);

    let obligation = Obligation::with_depth(cause.clone(), depth, projection_ty.clone());
    match project_type(selcx, &obligation) {
        Ok(ProjectedTy::Progress(projected_ty, mut obligations)) => {
            // if projection succeeded, then what we get out of this
            // is also non-normalized (consider: it was derived from
            // an impl, where-clause etc) and hence we must
            // re-normalize it

            debug!("normalize_projection_type: projected_ty={:?} depth={} obligations={:?}",
                   projected_ty,
                   depth,
                   obligations);

            if projected_ty.has_projection_types() {
                let mut normalizer = AssociatedTypeNormalizer::new(selcx, cause, depth+1);
                let normalized_ty = normalizer.fold(&projected_ty);

                debug!("normalize_projection_type: normalized_ty={:?} depth={}",
                       normalized_ty,
                       depth);

                obligations.extend(normalizer.obligations);
                Some(Normalized {
                    value: normalized_ty,
                    obligations: obligations,
                })
            } else {
                Some(Normalized {
                    value: projected_ty,
                    obligations: obligations,
                })
            }
        }
        Ok(ProjectedTy::NoProgress(projected_ty)) => {
            debug!("normalize_projection_type: projected_ty={:?} no progress",
                   projected_ty);
            Some(Normalized {
                value: projected_ty,
                obligations: vec!()
            })
        }
        Err(ProjectionTyError::TooManyCandidates) => {
            debug!("normalize_projection_type: too many candidates");
            None
        }
        Err(ProjectionTyError::TraitSelectionError(_)) => {
            debug!("normalize_projection_type: ERROR");
            // if we got an error processing the `T as Trait` part,
            // just return `ty::err` but add the obligation `T :
            // Trait`, which when processed will cause the error to be
            // reported later

            Some(normalize_to_error(selcx, projection_ty, cause, depth))
        }
    }
}

/// in various error cases, we just set TyError and return an obligation
/// that, when fulfilled, will lead to an error.
///
/// FIXME: the TyError created here can enter the obligation we create,
/// leading to error messages involving TyError.
fn normalize_to_error<'a,'tcx>(selcx: &mut SelectionContext<'a,'tcx>,
                               projection_ty: ty::ProjectionTy<'tcx>,
                               cause: ObligationCause<'tcx>,
                               depth: usize)
                               -> NormalizedTy<'tcx>
{
    let trait_ref = projection_ty.trait_ref.to_poly_trait_ref();
    let trait_obligation = Obligation { cause: cause,
                                        recursion_depth: depth,
                                        predicate: trait_ref.to_predicate() };
    Normalized {
        value: selcx.tcx().types.err,
        obligations: vec!(trait_obligation)
    }
}

enum ProjectedTy<'tcx> {
    Progress(Ty<'tcx>, Vec<PredicateObligation<'tcx>>),
    NoProgress(Ty<'tcx>),
}

/// Compute the result of a projection type (if we can).
fn project_type<'cx,'tcx>(
    selcx: &mut SelectionContext<'cx,'tcx>,
    obligation: &ProjectionTyObligation<'tcx>)
    -> Result<ProjectedTy<'tcx>, ProjectionTyError<'tcx>>
{
    debug!("project(obligation={:?})",
           obligation);

    let recursion_limit = selcx.tcx().sess.recursion_limit.get();
    if obligation.recursion_depth >= recursion_limit {
        debug!("project: overflow!");
        report_overflow_error(selcx.infcx(), &obligation);
    }

    let obligation_trait_ref =
        selcx.infcx().resolve_type_vars_if_possible(&obligation.predicate.trait_ref);

    debug!("project: obligation_trait_ref={:?}", obligation_trait_ref);

    if obligation_trait_ref.references_error() {
        return Ok(ProjectedTy::Progress(selcx.tcx().types.err, vec!()));
    }

    let mut candidates = ProjectionTyCandidateSet {
        vec: Vec::new(),
        ambiguous: false,
    };

    assemble_candidates_from_param_env(selcx,
                                       obligation,
                                       &obligation_trait_ref,
                                       &mut candidates);

    assemble_candidates_from_trait_def(selcx,
                                       obligation,
                                       &obligation_trait_ref,
                                       &mut candidates);

    if let Err(e) = assemble_candidates_from_impls(selcx,
                                                   obligation,
                                                   &obligation_trait_ref,
                                                   &mut candidates) {
        return Err(ProjectionTyError::TraitSelectionError(e));
    }

    debug!("{} candidates, ambiguous={}",
           candidates.vec.len(),
           candidates.ambiguous);

    // Inherent ambiguity that prevents us from even enumerating the
    // candidates.
    if candidates.ambiguous {
        return Err(ProjectionTyError::TooManyCandidates);
    }

    // Drop duplicates.
    //
    // Note: `candidates.vec` seems to be on the critical path of the
    // compiler. Replacing it with an hash set was also tried, which would
    // render the following dedup unnecessary. It led to cleaner code but
    // prolonged compiling time of `librustc` from 5m30s to 6m in one test, or
    // ~9% performance lost.
    if candidates.vec.len() > 1 {
        let mut i = 0;
        while i < candidates.vec.len() {
            let has_dup = (0..i).any(|j| candidates.vec[i] == candidates.vec[j]);
            if has_dup {
                candidates.vec.swap_remove(i);
            } else {
                i += 1;
            }
        }
    }

    // Prefer where-clauses. As in select, if there are multiple
    // candidates, we prefer where-clause candidates over impls.  This
    // may seem a bit surprising, since impls are the source of
    // "truth" in some sense, but in fact some of the impls that SEEM
    // applicable are not, because of nested obligations. Where
    // clauses are the safer choice. See the comment on
    // `select::SelectionCandidate` and #21974 for more details.
    if candidates.vec.len() > 1 {
        debug!("retaining param-env candidates only from {:?}", candidates.vec);
        candidates.vec.retain(|c| match *c {
            ProjectionTyCandidate::ParamEnv(..) => true,
            ProjectionTyCandidate::Impl(..) |
            ProjectionTyCandidate::Closure(..) |
            ProjectionTyCandidate::TraitDef(..) |
            ProjectionTyCandidate::FnPointer(..) => false,
        });
        debug!("resulting candidate set: {:?}", candidates.vec);
        if candidates.vec.len() != 1 {
            return Err(ProjectionTyError::TooManyCandidates);
        }
    }

    assert!(candidates.vec.len() <= 1);

    match candidates.vec.pop() {
        Some(candidate) => {
            let (ty, obligations) = confirm_candidate(selcx, obligation, candidate);
            Ok(ProjectedTy::Progress(ty, obligations))
        }
        None => {
            Ok(ProjectedTy::NoProgress(selcx.tcx().mk_projection(
                obligation.predicate.trait_ref.clone(),
                obligation.predicate.item_name)))
        }
    }
}

/// The first thing we have to do is scan through the parameter
/// environment to see whether there are any projection predicates
/// there that can answer this question.
fn assemble_candidates_from_param_env<'cx,'tcx>(
    selcx: &mut SelectionContext<'cx,'tcx>,
    obligation: &ProjectionTyObligation<'tcx>,
    obligation_trait_ref: &ty::TraitRef<'tcx>,
    candidate_set: &mut ProjectionTyCandidateSet<'tcx>)
{
    debug!("assemble_candidates_from_param_env(..)");
    let env_predicates = selcx.param_env().caller_bounds.iter().cloned();
    assemble_candidates_from_predicates(selcx,
                                        obligation,
                                        obligation_trait_ref,
                                        candidate_set,
                                        ProjectionTyCandidate::ParamEnv,
                                        env_predicates);
}

/// In the case of a nested projection like <<A as Foo>::FooT as Bar>::BarT, we may find
/// that the definition of `Foo` has some clues:
///
/// ```
/// trait Foo {
///     type FooT : Bar<BarT=i32>
/// }
/// ```
///
/// Here, for example, we could conclude that the result is `i32`.
fn assemble_candidates_from_trait_def<'cx,'tcx>(
    selcx: &mut SelectionContext<'cx,'tcx>,
    obligation: &ProjectionTyObligation<'tcx>,
    obligation_trait_ref: &ty::TraitRef<'tcx>,
    candidate_set: &mut ProjectionTyCandidateSet<'tcx>)
{
    debug!("assemble_candidates_from_trait_def(..)");

    // Check whether the self-type is itself a projection.
    let trait_ref = match obligation_trait_ref.self_ty().sty {
        ty::TyProjection(ref data) => data.trait_ref.clone(),
        ty::TyInfer(ty::TyVar(_)) => {
            // If the self-type is an inference variable, then it MAY wind up
            // being a projected type, so induce an ambiguity.
            candidate_set.ambiguous = true;
            return;
        }
        _ => { return; }
    };

    // If so, extract what we know from the trait and try to come up with a good answer.
    let trait_predicates = selcx.tcx().lookup_predicates(trait_ref.def_id);
    let bounds = trait_predicates.instantiate(selcx.tcx(), trait_ref.substs);
    let bounds = elaborate_predicates(selcx.tcx(), bounds.predicates.into_vec());
    assemble_candidates_from_predicates(selcx,
                                        obligation,
                                        obligation_trait_ref,
                                        candidate_set,
                                        ProjectionTyCandidate::TraitDef,
                                        bounds)
}

fn assemble_candidates_from_predicates<'cx,'tcx,I>(
    selcx: &mut SelectionContext<'cx,'tcx>,
    obligation: &ProjectionTyObligation<'tcx>,
    obligation_trait_ref: &ty::TraitRef<'tcx>,
    candidate_set: &mut ProjectionTyCandidateSet<'tcx>,
    ctor: fn(ty::PolyProjectionPredicate<'tcx>) -> ProjectionTyCandidate<'tcx>,
    env_predicates: I)
    where I: Iterator<Item=ty::Predicate<'tcx>>
{
    debug!("assemble_candidates_from_predicates(obligation={:?})",
           obligation);
    let infcx = selcx.infcx();
    for predicate in env_predicates {
        debug!("assemble_candidates_from_predicates: predicate={:?}",
               predicate);
        match predicate {
            ty::Predicate::Projection(ref data) => {
                let same_name = data.item_name() == obligation.predicate.item_name;

                let is_match = same_name && infcx.probe(|_| {
                    let origin = TypeOrigin::Misc(obligation.cause.span);
                    let data_poly_trait_ref =
                        data.to_poly_trait_ref();
                    let obligation_poly_trait_ref =
                        obligation_trait_ref.to_poly_trait_ref();
                    infcx.sub_poly_trait_refs(false,
                                              origin,
                                              data_poly_trait_ref,
                                              obligation_poly_trait_ref).is_ok()
                });

                debug!("assemble_candidates_from_predicates: candidate={:?} \
                                                             is_match={} same_name={}",
                       data, is_match, same_name);

                if is_match {
                    candidate_set.vec.push(ctor(data.clone()));
                }
            }
            _ => { }
        }
    }
}

fn assemble_candidates_from_object_type<'cx,'tcx>(
    selcx: &mut SelectionContext<'cx,'tcx>,
    obligation:  &ProjectionTyObligation<'tcx>,
    obligation_trait_ref: &ty::TraitRef<'tcx>,
    candidate_set: &mut ProjectionTyCandidateSet<'tcx>)
{
    let self_ty = obligation_trait_ref.self_ty();
    let object_ty = selcx.infcx().shallow_resolve(self_ty);
    debug!("assemble_candidates_from_object_type(object_ty={:?})",
           object_ty);
    let data = match object_ty.sty {
        ty::TyTrait(ref data) => data,
        _ => {
            selcx.tcx().sess.span_bug(
                obligation.cause.span,
                &format!("assemble_candidates_from_object_type called with non-object: {:?}",
                         object_ty));
        }
    };
    let projection_bounds = data.projection_bounds_with_self_ty(selcx.tcx(), object_ty);
    let env_predicates = projection_bounds.iter()
                                          .map(|p| p.to_predicate())
                                          .collect();
    let env_predicates = elaborate_predicates(selcx.tcx(), env_predicates);
    assemble_candidates_from_predicates(selcx,
                                        obligation,
                                        obligation_trait_ref,
                                        candidate_set,
                                        ProjectionTyCandidate::ParamEnv,
                                        env_predicates)
}

fn assemble_candidates_from_impls<'cx,'tcx>(
    selcx: &mut SelectionContext<'cx,'tcx>,
    obligation: &ProjectionTyObligation<'tcx>,
    obligation_trait_ref: &ty::TraitRef<'tcx>,
    candidate_set: &mut ProjectionTyCandidateSet<'tcx>)
    -> Result<(), SelectionError<'tcx>>
{
    // If we are resolving `<T as TraitRef<...>>::Item == Type`,
    // start out by selecting the predicate `T as TraitRef<...>`:
    let poly_trait_ref = obligation_trait_ref.to_poly_trait_ref();
    let trait_obligation = obligation.with(poly_trait_ref.to_poly_trait_predicate());
    let vtable = match selcx.select(&trait_obligation) {
        Ok(Some(vtable)) => vtable,
        Ok(None) => {
            candidate_set.ambiguous = true;
            return Ok(());
        }
        Err(e) => {
            debug!("assemble_candidates_from_impls: selection error {:?}",
                   e);
            return Err(e);
        }
    };

    match vtable {
        super::VtableImpl(data) => {
            debug!("assemble_candidates_from_impls: impl candidate {:?}",
                   data);

            candidate_set.vec.push(
                ProjectionTyCandidate::Impl(data));
        }
        super::VtableObject(_) => {
            assemble_candidates_from_object_type(
                selcx, obligation, obligation_trait_ref, candidate_set);
        }
        super::VtableClosure(data) => {
            candidate_set.vec.push(
                ProjectionTyCandidate::Closure(data));
        }
        super::VtableFnPointer(fn_type) => {
            candidate_set.vec.push(
                ProjectionTyCandidate::FnPointer(fn_type));
        }
        super::VtableParam(..) => {
            // This case tell us nothing about the value of an
            // associated type. Consider:
            //
            // ```
            // trait SomeTrait { type Foo; }
            // fn foo<T:SomeTrait>(...) { }
            // ```
            //
            // If the user writes `<T as SomeTrait>::Foo`, then the `T
            // : SomeTrait` binding does not help us decide what the
            // type `Foo` is (at least, not more specifically than
            // what we already knew).
            //
            // But wait, you say! What about an example like this:
            //
            // ```
            // fn bar<T:SomeTrait<Foo=usize>>(...) { ... }
            // ```
            //
            // Doesn't the `T : Sometrait<Foo=usize>` predicate help
            // resolve `T::Foo`? And of course it does, but in fact
            // that single predicate is desugared into two predicates
            // in the compiler: a trait predicate (`T : SomeTrait`) and a
            // projection. And the projection where clause is handled
            // in `assemble_candidates_from_param_env`.
        }
        super::VtableDefaultImpl(..) |
        super::VtableBuiltin(..) => {
            // These traits have no associated types.
            selcx.tcx().sess.span_bug(
                obligation.cause.span,
                &format!("Cannot project an associated type from `{:?}`",
                         vtable));
        }
    }

    Ok(())
}

fn confirm_candidate<'cx,'tcx>(
    selcx: &mut SelectionContext<'cx,'tcx>,
    obligation: &ProjectionTyObligation<'tcx>,
    candidate: ProjectionTyCandidate<'tcx>)
    -> (Ty<'tcx>, Vec<PredicateObligation<'tcx>>)
{
    debug!("confirm_candidate(candidate={:?}, obligation={:?})",
           candidate,
           obligation);

    match candidate {
        ProjectionTyCandidate::ParamEnv(poly_projection) |
        ProjectionTyCandidate::TraitDef(poly_projection) => {
            confirm_param_env_candidate(selcx, obligation, poly_projection)
        }

        ProjectionTyCandidate::Impl(impl_vtable) => {
            confirm_impl_candidate(selcx, obligation, impl_vtable)
        }

        ProjectionTyCandidate::Closure(closure_vtable) => {
            confirm_closure_candidate(selcx, obligation, closure_vtable)
        }

        ProjectionTyCandidate::FnPointer(fn_type) => {
            confirm_fn_pointer_candidate(selcx, obligation, fn_type)
        }
    }
}

fn confirm_fn_pointer_candidate<'cx,'tcx>(
    selcx: &mut SelectionContext<'cx,'tcx>,
    obligation: &ProjectionTyObligation<'tcx>,
    fn_type: Ty<'tcx>)
    -> (Ty<'tcx>, Vec<PredicateObligation<'tcx>>)
{
    let fn_type = selcx.infcx().shallow_resolve(fn_type);
    let sig = fn_type.fn_sig();
    confirm_callable_candidate(selcx, obligation, sig, util::TupleArgumentsFlag::Yes)
}

fn confirm_closure_candidate<'cx,'tcx>(
    selcx: &mut SelectionContext<'cx,'tcx>,
    obligation: &ProjectionTyObligation<'tcx>,
    vtable: VtableClosureData<'tcx, PredicateObligation<'tcx>>)
    -> (Ty<'tcx>, Vec<PredicateObligation<'tcx>>)
{
    let closure_typer = selcx.closure_typer();
    let closure_type = closure_typer.closure_type(vtable.closure_def_id, &vtable.substs);
    let Normalized {
        value: closure_type,
        mut obligations
    } = normalize_with_depth(selcx,
                             obligation.cause.clone(),
                             obligation.recursion_depth+1,
                             &closure_type);
    let (ty, mut cc_obligations) = confirm_callable_candidate(selcx,
                                                              obligation,
                                                              &closure_type.sig,
                                                              util::TupleArgumentsFlag::No);
    obligations.append(&mut cc_obligations);
    (ty, obligations)
}

fn confirm_callable_candidate<'cx,'tcx>(
    selcx: &mut SelectionContext<'cx,'tcx>,
    obligation: &ProjectionTyObligation<'tcx>,
    fn_sig: &ty::PolyFnSig<'tcx>,
    flag: util::TupleArgumentsFlag)
    -> (Ty<'tcx>, Vec<PredicateObligation<'tcx>>)
{
    let tcx = selcx.tcx();

    debug!("confirm_callable_candidate({:?},{:?})",
           obligation,
           fn_sig);

    // the `Output` associated type is declared on `FnOnce`
    let fn_once_def_id = tcx.lang_items.fn_once_trait().unwrap();

    // Note: we unwrap the binder here but re-create it below (1)
    let ty::Binder((trait_ref, ret_type)) =
        util::closure_trait_ref_and_return_type(tcx,
                                                fn_once_def_id,
                                                obligation.predicate.trait_ref.self_ty(),
                                                fn_sig,
                                                flag);

    let predicate = ty::Binder(ty::ProjectionPredicate { // (1) recreate binder here
        projection_ty: ty::ProjectionTy {
            trait_ref: trait_ref,
            item_name: token::intern(FN_OUTPUT_NAME),
        },
        ty: ret_type
    });

    confirm_param_env_candidate(selcx, obligation, predicate)
}

fn confirm_param_env_candidate<'cx,'tcx>(
    selcx: &mut SelectionContext<'cx,'tcx>,
    obligation: &ProjectionTyObligation<'tcx>,
    poly_projection: ty::PolyProjectionPredicate<'tcx>)
    -> (Ty<'tcx>, Vec<PredicateObligation<'tcx>>)
{
    let infcx = selcx.infcx();

    let projection =
        infcx.replace_late_bound_regions_with_fresh_var(
            obligation.cause.span,
            infer::LateBoundRegionConversionTime::HigherRankedType,
            &poly_projection).0;

    assert_eq!(projection.projection_ty.item_name,
               obligation.predicate.item_name);

    let origin = TypeOrigin::RelateOutputImplTypes(obligation.cause.span);
    match infcx.eq_trait_refs(false,
                              origin,
                              obligation.predicate.trait_ref.clone(),
                              projection.projection_ty.trait_ref.clone()) {
        Ok(()) => { }
        Err(e) => {
            selcx.tcx().sess.span_bug(
                obligation.cause.span,
                &format!("Failed to unify `{:?}` and `{:?}` in projection: {}",
                         obligation,
                         projection,
                         e));
        }
    }

    (projection.ty, vec!())
}

fn confirm_impl_candidate<'cx,'tcx>(
    selcx: &mut SelectionContext<'cx,'tcx>,
    obligation: &ProjectionTyObligation<'tcx>,
    impl_vtable: VtableImplData<'tcx, PredicateObligation<'tcx>>)
    -> (Ty<'tcx>, Vec<PredicateObligation<'tcx>>)
{
    // there don't seem to be nicer accessors to these:
    let impl_or_trait_items_map = selcx.tcx().impl_or_trait_items.borrow();

    // Look for the associated type in the impl
    for impl_item in &selcx.tcx().impl_items.borrow()[&impl_vtable.impl_def_id] {
        if let ty::TypeTraitItem(ref assoc_ty) = impl_or_trait_items_map[&impl_item.def_id()] {
            if assoc_ty.name == obligation.predicate.item_name {
                return (assoc_ty.ty.unwrap().subst(selcx.tcx(), &impl_vtable.substs),
                        impl_vtable.nested);
            }
        }
    }

    // It is not in the impl - get the default from the trait.
    let trait_ref = obligation.predicate.trait_ref;
    for trait_item in selcx.tcx().trait_items(trait_ref.def_id).iter() {
        if let &ty::TypeTraitItem(ref assoc_ty) = trait_item {
            if assoc_ty.name == obligation.predicate.item_name {
                if let Some(ty) = assoc_ty.ty {
                    return (ty.subst(selcx.tcx(), trait_ref.substs),
                            impl_vtable.nested);
                } else {
                    // This means that the impl is missing a
                    // definition for the associated type. This error
                    // ought to be reported by the type checker method
                    // `check_impl_items_against_trait`, so here we
                    // just return TyError.
                    debug!("confirm_impl_candidate: no associated type {:?} for {:?}",
                           assoc_ty.name,
                           trait_ref);
                    return (selcx.tcx().types.err, vec!());
                }
            }
        }
    }

    selcx.tcx().sess.span_bug(obligation.cause.span,
                              &format!("No associated type for {:?}",
                                       trait_ref));
}
