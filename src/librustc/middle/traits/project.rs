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
use super::Obligation;
use super::ObligationCause;
use super::Overflow;
use super::PredicateObligation;
use super::SelectionContext;
use super::SelectionError;
use super::VtableImplData;
use super::util;

use middle::infer;
use middle::subst::{Subst, Substs};
use middle::ty::{self, AsPredicate, ReferencesError, RegionEscape,
                 HasProjectionTypes, ToPolyTraitRef, Ty};
use middle::ty_fold::{self, TypeFoldable, TypeFolder};
use std::rc::Rc;
use syntax::ast;
use syntax::parse::token;
use util::common::FN_OUTPUT_NAME;
use util::ppaux::Repr;

pub type PolyProjectionObligation<'tcx> =
    Obligation<'tcx, ty::PolyProjectionPredicate<'tcx>>;

pub type ProjectionObligation<'tcx> =
    Obligation<'tcx, ty::ProjectionPredicate<'tcx>>;

pub type ProjectionTyObligation<'tcx> =
    Obligation<'tcx, ty::ProjectionTy<'tcx>>;

/// When attempting to resolve `<T as TraitRef>::Name` ...
pub enum ProjectionTyError<'tcx> {
    /// ...we found multiple sources of information and couldn't resolve the ambiguity.
    TooManyCandidates,

    /// ...an error occurred matching `T : TraitRef`
    TraitSelectionError(SelectionError<'tcx>),
}

#[derive(Clone)]
pub struct MismatchedProjectionTypes<'tcx> {
    pub err: ty::type_err<'tcx>
}

#[derive(PartialEq, Eq)]
enum ProjectionTyCandidate<'tcx> {
    ParamEnv(ty::PolyProjectionPredicate<'tcx>),
    Impl(VtableImplData<'tcx, PredicateObligation<'tcx>>),
    Closure(ast::DefId, Substs<'tcx>),
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
    debug!("poly_project_and_unify_type(obligation={})",
           obligation.repr(selcx.tcx()));

    let infcx = selcx.infcx();
    infcx.try(|snapshot| {
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
    debug!("project_and_unify_type(obligation={})",
           obligation.repr(selcx.tcx()));

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

    debug!("project_and_unify_type: normalized_ty={} obligations={}",
           normalized_ty.repr(selcx.tcx()),
           obligations.repr(selcx.tcx()));

    let infcx = selcx.infcx();
    let origin = infer::RelateOutputImplTypes(obligation.cause.span);
    match infer::mk_eqty(infcx, true, origin, normalized_ty, obligation.predicate.ty) {
        Ok(()) => Ok(Some(obligations)),
        Err(err) => Err(MismatchedProjectionTypes { err: err }),
    }
}

fn consider_unification_despite_ambiguity<'cx,'tcx>(selcx: &mut SelectionContext<'cx,'tcx>,
                                                    obligation: &ProjectionObligation<'tcx>) {
    debug!("consider_unification_despite_ambiguity(obligation={})",
           obligation.repr(selcx.tcx()));

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
        ty::ty_closure(closure_def_id, substs) => {
            let closure_typer = selcx.closure_typer();
            let closure_type = closure_typer.closure_type(closure_def_id, substs);
            let ty::Binder((_, ret_type)) =
                util::closure_trait_ref_and_return_type(infcx.tcx,
                                                        def_id,
                                                        self_ty,
                                                        &closure_type.sig,
                                                        util::TupleArgumentsFlag::No);
            let (ret_type, _) =
                infcx.replace_late_bound_regions_with_fresh_var(
                    obligation.cause.span,
                    infer::AssocTypeProjection(obligation.predicate.projection_ty.item_name),
                    &ty::Binder(ret_type));
            debug!("consider_unification_despite_ambiguity: ret_type={:?}",
                   ret_type.repr(selcx.tcx()));
            let origin = infer::RelateOutputImplTypes(obligation.cause.span);
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
    where T : TypeFoldable<'tcx> + HasProjectionTypes + Clone + Repr<'tcx>
{
    normalize_with_depth(selcx, cause, 0, value)
}

/// As `normalize`, but with a custom depth.
pub fn normalize_with_depth<'a,'b,'tcx,T>(selcx: &'a mut SelectionContext<'b,'tcx>,
                                          cause: ObligationCause<'tcx>,
                                          depth: uint,
                                          value: &T)
                                          -> Normalized<'tcx, T>
    where T : TypeFoldable<'tcx> + HasProjectionTypes + Clone + Repr<'tcx>
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
    depth: uint,
}

impl<'a,'b,'tcx> AssociatedTypeNormalizer<'a,'b,'tcx> {
    fn new(selcx: &'a mut SelectionContext<'b,'tcx>,
           cause: ObligationCause<'tcx>,
           depth: uint)
           -> AssociatedTypeNormalizer<'a,'b,'tcx>
    {
        AssociatedTypeNormalizer {
            selcx: selcx,
            cause: cause,
            obligations: vec!(),
            depth: depth,
        }
    }

    fn fold<T:TypeFoldable<'tcx> + HasProjectionTypes + Clone>(&mut self, value: &T) -> T {
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

        let ty = ty_fold::super_fold_ty(self, ty);
        match ty.sty {
            ty::ty_projection(ref data) if !data.has_escaping_regions() => { // (*)

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
                self.obligations.extend(obligations.into_iter());
                ty
            }

            _ => {
                ty
            }
        }
    }
}

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
    depth: uint)
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
            let obligation = Obligation::with_depth(cause, depth+1, projection.as_predicate());
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
    depth: uint)
    -> Option<NormalizedTy<'tcx>>
{
    debug!("normalize_projection_type(\
           projection_ty={}, \
           depth={})",
           projection_ty.repr(selcx.tcx()),
           depth);

    let obligation = Obligation::with_depth(cause.clone(), depth, projection_ty.clone());
    match project_type(selcx, &obligation) {
        Ok(ProjectedTy::Progress(projected_ty, mut obligations)) => {
            // if projection succeeded, then what we get out of this
            // is also non-normalized (consider: it was derived from
            // an impl, where-clause etc) and hence we must
            // re-normalize it

            debug!("normalize_projection_type: projected_ty={} depth={} obligations={}",
                   projected_ty.repr(selcx.tcx()),
                   depth,
                   obligations.repr(selcx.tcx()));

            if ty::type_has_projection(projected_ty) {
                let tcx = selcx.tcx();
                let mut normalizer = AssociatedTypeNormalizer::new(selcx, cause, depth);
                let normalized_ty = normalizer.fold(&projected_ty);

                debug!("normalize_projection_type: normalized_ty={} depth={}",
                       normalized_ty.repr(tcx),
                       depth);

                obligations.extend(normalizer.obligations.into_iter());
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
            Some(Normalized {
                value: projected_ty,
                obligations: vec!()
            })
        }
        Err(ProjectionTyError::TooManyCandidates) => {
            None
        }
        Err(ProjectionTyError::TraitSelectionError(_)) => {
            // if we got an error processing the `T as Trait` part,
            // just return `ty::err` but add the obligation `T :
            // Trait`, which when processed will cause the error to be
            // reported later

            Some(normalize_to_error(selcx, projection_ty, cause, depth))
        }
    }
}

/// in various error cases, we just set ty_err and return an obligation
/// that, when fulfilled, will lead to an error
fn normalize_to_error<'a,'tcx>(selcx: &mut SelectionContext<'a,'tcx>,
                               projection_ty: ty::ProjectionTy<'tcx>,
                               cause: ObligationCause<'tcx>,
                               depth: uint)
                               -> NormalizedTy<'tcx>
{
    let trait_ref = projection_ty.trait_ref.to_poly_trait_ref();
    let trait_obligation = Obligation { cause: cause,
                                        recursion_depth: depth,
                                        predicate: trait_ref.as_predicate() };
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
    debug!("project(obligation={})",
           obligation.repr(selcx.tcx()));

    let recursion_limit = selcx.tcx().sess.recursion_limit.get();
    if obligation.recursion_depth >= recursion_limit {
        debug!("project: overflow!");
        return Err(ProjectionTyError::TraitSelectionError(Overflow));
    }

    let obligation_trait_ref =
        selcx.infcx().resolve_type_vars_if_possible(&obligation.predicate.trait_ref);

    debug!("project: obligation_trait_ref={}", obligation_trait_ref.repr(selcx.tcx()));

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

    // We probably need some winnowing logic similar to select here.

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

    if candidates.ambiguous || candidates.vec.len() > 1 {
        return Err(ProjectionTyError::TooManyCandidates);
    }

    match candidates.vec.pop() {
        Some(candidate) => {
            let (ty, obligations) = confirm_candidate(selcx, obligation, candidate);
            Ok(ProjectedTy::Progress(ty, obligations))
        }
        None => {
            Ok(ProjectedTy::NoProgress(ty::mk_projection(selcx.tcx(),
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
    obligation_trait_ref: &Rc<ty::TraitRef<'tcx>>,
    candidate_set: &mut ProjectionTyCandidateSet<'tcx>)
{
    let env_predicates = selcx.param_env().caller_bounds.clone();
    assemble_candidates_from_predicates(selcx, obligation, obligation_trait_ref,
                                        candidate_set, env_predicates);
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
    obligation_trait_ref: &Rc<ty::TraitRef<'tcx>>,
    candidate_set: &mut ProjectionTyCandidateSet<'tcx>)
{
    // Check whether the self-type is itself a projection.
    let trait_ref = match obligation_trait_ref.self_ty().sty {
        ty::ty_projection(ref data) => data.trait_ref.clone(),
        ty::ty_infer(ty::TyVar(_)) => {
            // If the self-type is an inference variable, then it MAY wind up
            // being a projected type, so induce an ambiguity.
            candidate_set.ambiguous = true;
            return;
        }
        _ => { return; }
    };

    // If so, extract what we know from the trait and try to come up with a good answer.
    let trait_predicates = ty::lookup_predicates(selcx.tcx(), trait_ref.def_id);
    let bounds = trait_predicates.instantiate(selcx.tcx(), trait_ref.substs);
    assemble_candidates_from_predicates(selcx, obligation, obligation_trait_ref,
                                        candidate_set, bounds.predicates.into_vec());
}

fn assemble_candidates_from_predicates<'cx,'tcx>(
    selcx: &mut SelectionContext<'cx,'tcx>,
    obligation: &ProjectionTyObligation<'tcx>,
    obligation_trait_ref: &Rc<ty::TraitRef<'tcx>>,
    candidate_set: &mut ProjectionTyCandidateSet<'tcx>,
    env_predicates: Vec<ty::Predicate<'tcx>>)
{
    debug!("assemble_candidates_from_predicates(obligation={}, env_predicates={})",
           obligation.repr(selcx.tcx()),
           env_predicates.repr(selcx.tcx()));
    let infcx = selcx.infcx();
    for predicate in elaborate_predicates(selcx.tcx(), env_predicates) {
        match predicate {
            ty::Predicate::Projection(ref data) => {
                let same_name = data.item_name() == obligation.predicate.item_name;

                let is_match = same_name && infcx.probe(|_| {
                    let origin = infer::Misc(obligation.cause.span);
                    let data_poly_trait_ref =
                        data.to_poly_trait_ref();
                    let obligation_poly_trait_ref =
                        obligation_trait_ref.to_poly_trait_ref();
                    infcx.sub_poly_trait_refs(false,
                                              origin,
                                              data_poly_trait_ref,
                                              obligation_poly_trait_ref).is_ok()
                });

                debug!("assemble_candidates_from_predicates: candidate {} is_match {} same_name {}",
                       data.repr(selcx.tcx()),
                       is_match,
                       same_name);

                if is_match {
                    candidate_set.vec.push(
                        ProjectionTyCandidate::ParamEnv(data.clone()));
                }
            }
            _ => { }
        }
    }
}

fn assemble_candidates_from_object_type<'cx,'tcx>(
    selcx: &mut SelectionContext<'cx,'tcx>,
    obligation:  &ProjectionTyObligation<'tcx>,
    obligation_trait_ref: &Rc<ty::TraitRef<'tcx>>,
    candidate_set: &mut ProjectionTyCandidateSet<'tcx>,
    object_ty: Ty<'tcx>)
{
    let infcx = selcx.infcx();
    debug!("assemble_candidates_from_object_type(object_ty={})",
           object_ty.repr(infcx.tcx));
    let data = match object_ty.sty {
        ty::ty_trait(ref data) => data,
        _ => {
            selcx.tcx().sess.span_bug(
                obligation.cause.span,
                &format!("assemble_candidates_from_object_type called with non-object: {}",
                         object_ty.repr(selcx.tcx())));
        }
    };
    let projection_bounds = data.projection_bounds_with_self_ty(selcx.tcx(), object_ty);
    let env_predicates = projection_bounds.iter()
                                          .map(|p| p.as_predicate())
                                          .collect();
    assemble_candidates_from_predicates(selcx, obligation, obligation_trait_ref,
                                        candidate_set, env_predicates)
}

fn assemble_candidates_from_impls<'cx,'tcx>(
    selcx: &mut SelectionContext<'cx,'tcx>,
    obligation: &ProjectionTyObligation<'tcx>,
    obligation_trait_ref: &Rc<ty::TraitRef<'tcx>>,
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
            debug!("assemble_candidates_from_impls: selection error {}",
                   e.repr(selcx.tcx()));
            return Err(e);
        }
    };

    match vtable {
        super::VtableImpl(data) => {
            debug!("assemble_candidates_from_impls: impl candidate {}",
                   data.repr(selcx.tcx()));

            candidate_set.vec.push(
                ProjectionTyCandidate::Impl(data));
        }
        super::VtableObject(data) => {
            assemble_candidates_from_object_type(
                selcx, obligation, obligation_trait_ref, candidate_set,
                data.object_ty);
        }
        super::VtableClosure(closure_def_id, substs) => {
            candidate_set.vec.push(
                ProjectionTyCandidate::Closure(closure_def_id, substs));
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
            // fn bar<T:SomeTrait<Foo=uint>>(...) { ... }
            // ```
            //
            // Doesn't the `T : Sometrait<Foo=uint>` predicate help
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
                &format!("Cannot project an associated type from `{}`",
                         vtable.repr(selcx.tcx())));
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
    let infcx = selcx.infcx();

    debug!("confirm_candidate(candidate={}, obligation={})",
           candidate.repr(infcx.tcx),
           obligation.repr(infcx.tcx));

    match candidate {
        ProjectionTyCandidate::ParamEnv(poly_projection) => {
            confirm_param_env_candidate(selcx, obligation, poly_projection)
        }

        ProjectionTyCandidate::Impl(impl_vtable) => {
            confirm_impl_candidate(selcx, obligation, impl_vtable)
        }

        ProjectionTyCandidate::Closure(def_id, substs) => {
            confirm_closure_candidate(selcx, obligation, def_id, &substs)
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
    let sig = ty::ty_fn_sig(fn_type);
    confirm_callable_candidate(selcx, obligation, sig, util::TupleArgumentsFlag::Yes)
}

fn confirm_closure_candidate<'cx,'tcx>(
    selcx: &mut SelectionContext<'cx,'tcx>,
    obligation: &ProjectionTyObligation<'tcx>,
    closure_def_id: ast::DefId,
    substs: &Substs<'tcx>)
    -> (Ty<'tcx>, Vec<PredicateObligation<'tcx>>)
{
    let closure_typer = selcx.closure_typer();
    let closure_type = closure_typer.closure_type(closure_def_id, substs);
    confirm_callable_candidate(selcx, obligation, &closure_type.sig, util::TupleArgumentsFlag::No)
}

fn confirm_callable_candidate<'cx,'tcx>(
    selcx: &mut SelectionContext<'cx,'tcx>,
    obligation: &ProjectionTyObligation<'tcx>,
    fn_sig: &ty::PolyFnSig<'tcx>,
    flag: util::TupleArgumentsFlag)
    -> (Ty<'tcx>, Vec<PredicateObligation<'tcx>>)
{
    let tcx = selcx.tcx();

    debug!("confirm_closure_candidate({},{})",
           obligation.repr(tcx),
           fn_sig.repr(tcx));

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

    let origin = infer::RelateOutputImplTypes(obligation.cause.span);
    match infcx.sub_trait_refs(false,
                               origin,
                               obligation.predicate.trait_ref.clone(),
                               projection.projection_ty.trait_ref.clone()) {
        Ok(()) => { }
        Err(e) => {
            selcx.tcx().sess.span_bug(
                obligation.cause.span,
                &format!("Failed to unify `{}` and `{}` in projection: {}",
                         obligation.repr(selcx.tcx()),
                         projection.repr(selcx.tcx()),
                         ty::type_err_to_str(selcx.tcx(), &e)));
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
    let impl_items_map = selcx.tcx().impl_items.borrow();
    let impl_or_trait_items_map = selcx.tcx().impl_or_trait_items.borrow();

    let impl_items = &impl_items_map[impl_vtable.impl_def_id];
    let mut impl_ty = None;
    for impl_item in impl_items {
        let assoc_type = match impl_or_trait_items_map[impl_item.def_id()] {
            ty::TypeTraitItem(ref assoc_type) => assoc_type.clone(),
            ty::MethodTraitItem(..) => { continue; }
        };

        if assoc_type.name != obligation.predicate.item_name {
            continue;
        }

        let impl_poly_ty = ty::lookup_item_type(selcx.tcx(), assoc_type.def_id);
        impl_ty = Some(impl_poly_ty.ty.subst(selcx.tcx(), &impl_vtable.substs));
        break;
    }

    match impl_ty {
        Some(ty) => (ty, impl_vtable.nested.into_vec()),
        None => {
            // This means that the impl is missing a
            // definition for the associated type. This error
            // ought to be reported by the type checker method
            // `check_impl_items_against_trait`, so here we
            // just return ty_err.
            (selcx.tcx().types.err, vec!())
        }
    }
}

impl<'tcx> Repr<'tcx> for ProjectionTyError<'tcx> {
    fn repr(&self, tcx: &ty::ctxt<'tcx>) -> String {
        match *self {
            ProjectionTyError::TooManyCandidates =>
                format!("NoCandidate"),
            ProjectionTyError::TraitSelectionError(ref e) =>
                format!("TraitSelectionError({})", e.repr(tcx)),
        }
    }
}

impl<'tcx> Repr<'tcx> for ProjectionTyCandidate<'tcx> {
    fn repr(&self, tcx: &ty::ctxt<'tcx>) -> String {
        match *self {
            ProjectionTyCandidate::ParamEnv(ref data) =>
                format!("ParamEnv({})", data.repr(tcx)),
            ProjectionTyCandidate::Impl(ref data) =>
                format!("Impl({})", data.repr(tcx)),
            ProjectionTyCandidate::Closure(ref a, ref b) =>
                format!("Closure(({},{}))", a.repr(tcx), b.repr(tcx)),
            ProjectionTyCandidate::FnPointer(a) =>
                format!("FnPointer(({}))", a.repr(tcx)),
        }
    }
}

impl<'tcx, T: TypeFoldable<'tcx>> TypeFoldable<'tcx> for Normalized<'tcx, T> {
    fn fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> Normalized<'tcx, T> {
        Normalized {
            value: self.value.fold_with(folder),
            obligations: self.obligations.fold_with(folder),
        }
    }
}

impl<'tcx, T:Repr<'tcx>> Repr<'tcx> for Normalized<'tcx, T> {
    fn repr(&self, tcx: &ty::ctxt<'tcx>) -> String {
        format!("Normalized({},{})",
                self.value.repr(tcx),
                self.obligations.repr(tcx))
    }
}
