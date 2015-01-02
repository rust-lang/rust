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

use middle::infer;
use middle::subst::Subst;
use middle::ty::{mod, AsPredicate, ReferencesError, RegionEscape,
                 HasProjectionTypes, ToPolyTraitRef, Ty};
use middle::ty_fold::{mod, TypeFoldable, TypeFolder};
use std::rc::Rc;
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

#[deriving(Clone)]
pub struct MismatchedProjectionTypes<'tcx> {
    pub err: ty::type_err<'tcx>
}

enum ProjectionTyCandidate<'tcx> {
    ParamEnv(ty::PolyProjectionPredicate<'tcx>),
    Impl(VtableImplData<'tcx, PredicateObligation<'tcx>>),
}

struct ProjectionTyCandidateSet<'tcx> {
    vec: Vec<ProjectionTyCandidate<'tcx>>,
    ambiguous: bool
}

pub fn poly_project_and_unify_type<'cx,'tcx>(
    selcx: &mut SelectionContext<'cx,'tcx>,
    obligation: &PolyProjectionObligation<'tcx>)
    -> Result<Option<Vec<PredicateObligation<'tcx>>>, MismatchedProjectionTypes<'tcx>>
{
    debug!("poly_project(obligation={})",
           obligation.repr(selcx.tcx()));

    let infcx = selcx.infcx();
    let result = infcx.try(|snapshot| {
        let (skol_predicate, skol_map) =
            infcx.skolemize_late_bound_regions(&obligation.predicate, snapshot);

        let skol_obligation = obligation.with(skol_predicate);
        match project_and_unify_type(selcx, &skol_obligation) {
            Ok(Some(obligations)) => {
                match infcx.leak_check(&skol_map, snapshot) {
                    Ok(()) => Ok(infcx.plug_leaks(skol_map, snapshot, &obligations)),
                    Err(e) => Err(Some(MismatchedProjectionTypes { err: e })),
                }
            }
            Ok(None) => {
                // Signal ambiguity using Err just so that infcx.try()
                // rolls back the snapshot. We adapt below.
                Err(None)
            }
            Err(e) => {
                Err(Some(e))
            }
        }
    });

    // Above, we use Err(None) to signal ambiguity so that the
    // snapshot will be rolled back. But here, we want to translate to
    // Ok(None). Kind of weird.
    match result {
        Ok(obligations) => Ok(Some(obligations)),
        Err(None) => Ok(None),
        Err(Some(e)) => Err(e),
    }
}

/// Compute result of projecting an associated type and unify it with
/// `obligation.predicate.ty` (if we can).
fn project_and_unify_type<'cx,'tcx>(
    selcx: &mut SelectionContext<'cx,'tcx>,
    obligation: &ProjectionObligation<'tcx>)
    -> Result<Option<Vec<PredicateObligation<'tcx>>>, MismatchedProjectionTypes<'tcx>>
{
    debug!("project_and_unify(obligation={})",
           obligation.repr(selcx.tcx()));

    let Normalized { value: normalized_ty, obligations } =
        match opt_normalize_projection_type(selcx,
                                            obligation.predicate.projection_ty.clone(),
                                            obligation.cause.clone(),
                                            obligation.recursion_depth) {
            Some(n) => n,
            None => { return Ok(None); }
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

pub fn normalize<'a,'b,'tcx,T>(selcx: &'a mut SelectionContext<'b,'tcx>,
                               cause: ObligationCause<'tcx>,
                               value: &T)
                               -> Normalized<'tcx, T>
    where T : TypeFoldable<'tcx> + HasProjectionTypes + Clone + Repr<'tcx>
{
    normalize_with_depth(selcx, cause, 0, value)
}

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
                ty_fold::super_fold_ty(self, ty)
            }
        }
    }
}

pub struct Normalized<'tcx,T> {
    pub value: T,
    pub obligations: Vec<PredicateObligation<'tcx>>,
}

pub type NormalizedTy<'tcx> = Normalized<'tcx, Ty<'tcx>>;

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
/// that, when fulfiled, will lead to an error
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

    assemble_candidates_from_object_type(selcx,
                                         obligation,
                                         &obligation_trait_ref,
                                         &mut candidates);

    if candidates.vec.is_empty() {
        assemble_candidates_from_param_env(selcx,
                                           obligation,
                                           &obligation_trait_ref,
                                           &mut candidates);

        if let Err(e) = assemble_candidates_from_impls(selcx,
                                                       obligation,
                                                       &obligation_trait_ref,
                                                       &mut candidates) {
            return Err(ProjectionTyError::TraitSelectionError(e));
        }
    }

    debug!("{} candidates, ambiguous={}",
           candidates.vec.len(),
           candidates.ambiguous);

    // We probably need some winnowing logic similar to select here.

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
    let env_predicates = selcx.param_env().caller_bounds.predicates.clone();
    let env_predicates = env_predicates.iter().cloned().collect();
    assemble_candidates_from_predicates(selcx, obligation, obligation_trait_ref,
                                        candidate_set, env_predicates);
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
                let is_match = infcx.probe(|_| {
                    let origin = infer::Misc(obligation.cause.span);
                    let obligation_poly_trait_ref =
                        obligation_trait_ref.to_poly_trait_ref();
                    let data_poly_trait_ref =
                        data.to_poly_trait_ref();
                    infcx.sub_poly_trait_refs(false,
                                              origin,
                                              obligation_poly_trait_ref,
                                              data_poly_trait_ref).is_ok()
                });

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
    obligation: &ProjectionTyObligation<'tcx>,
    obligation_trait_ref: &Rc<ty::TraitRef<'tcx>>,
    candidate_set: &mut ProjectionTyCandidateSet<'tcx>)
{
    let infcx = selcx.infcx();
    debug!("assemble_candidates_from_object_type(trait_ref={})",
           obligation_trait_ref.repr(infcx.tcx));
    let self_ty = obligation_trait_ref.self_ty();
    let data = match self_ty.sty {
        ty::ty_trait(ref data) => data,
        _ => { return; }
    };
    let projection_bounds = data.projection_bounds_with_self_ty(selcx.tcx(), self_ty);
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
            candidate_set.vec.push(
                ProjectionTyCandidate::Impl(data));
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
        super::VtableBuiltin(..) |
        super::VtableUnboxedClosure(..) |
        super::VtableFnPointer(..) => {
            // These traits have no associated types.
            selcx.tcx().sess.span_bug(
                obligation.cause.span,
                format!("Cannot project an associated type from `{}`",
                        vtable.repr(selcx.tcx())).as_slice());
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
                        format!("Failed to unify `{}` and `{}` in projection: {}",
                                obligation.repr(selcx.tcx()),
                                projection.repr(selcx.tcx()),
                                ty::type_err_to_str(selcx.tcx(), &e)).as_slice());
                }
            }

            (projection.ty, vec!())
        }

        ProjectionTyCandidate::Impl(impl_vtable) => {
            // there don't seem to be nicer accessors to these:
            let impl_items_map = selcx.tcx().impl_items.borrow();
            let impl_or_trait_items_map = selcx.tcx().impl_or_trait_items.borrow();

            let impl_items = &impl_items_map[impl_vtable.impl_def_id];
            let mut impl_ty = None;
            for impl_item in impl_items.iter() {
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
                Some(ty) => (ty, impl_vtable.nested.to_vec()),
                None => {
                    selcx.tcx().sess.span_bug(
                        obligation.cause.span,
                        format!("impl `{}` did not contain projection for `{}`",
                                impl_vtable.repr(selcx.tcx()),
                                obligation.repr(selcx.tcx())).as_slice());
                }
            }
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
                format!("Impl({})", data.repr(tcx))
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
