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
use super::PredicateObligation;
use super::SelectionContext;
use super::SelectionError;
use super::VtableImplData;

use middle::infer;
use middle::subst::Subst;
use middle::ty::{mod, AsPredicate, ToPolyTraitRef, Ty};
use util::ppaux::Repr;

pub type PolyProjectionObligation<'tcx> =
    Obligation<'tcx, ty::PolyProjectionPredicate<'tcx>>;

pub type ProjectionObligation<'tcx> =
    Obligation<'tcx, ty::ProjectionPredicate<'tcx>>;

pub type ProjectionTyObligation<'tcx> =
    Obligation<'tcx, ty::ProjectionTy<'tcx>>;

/// When attempting to resolve `<T as TraitRef>::Name == U`...
pub enum ProjectionError<'tcx> {
    /// ...we could not find any helpful information on what `Name`
    /// might be. This could occur, for example, if there is a where
    /// clause `T : TraitRef` but not `T : TraitRef<Name=V>`. When
    /// normalizing, this case is where we opt to normalize back to
    /// the projection type `<T as TraitRef>::Name`.
    NoCandidate,

    /// ...we found multiple sources of information and couldn't resolve the ambiguity.
    TooManyCandidates,

    /// ...`<T as TraitRef::Name>` ws resolved to some type `V` that failed to unify with `U`
    MismatchedTypes(MismatchedProjectionTypes<'tcx>),

    /// ...an error occurred matching `T : TraitRef`
    TraitSelectionError(SelectionError<'tcx>),
}

#[deriving(Clone)]
pub struct MismatchedProjectionTypes<'tcx> {
    pub err: ty::type_err<'tcx>
}

pub type ProjectionResult<'tcx, T> = Result<T, ProjectionError<'tcx>>;

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
    -> ProjectionResult<'tcx, ()>
{
    debug!("poly_project(obligation={})",
           obligation.repr(selcx.tcx()));

    let infcx = selcx.infcx();

    infcx.try(|snapshot| {
        let (skol_predicate, skol_map) =
            infcx.skolemize_late_bound_regions(&obligation.predicate, snapshot);

        let skol_obligation = obligation.with(skol_predicate);
        let () = try!(project_and_unify_type(selcx, &skol_obligation));
        match infcx.leak_check(&skol_map, snapshot) {
            Ok(()) => Ok(()),
            Err(e) => Err(ProjectionError::MismatchedTypes(MismatchedProjectionTypes{err: e})),
        }
    })
}

/// Compute result of projecting an associated type and unify it with
/// `obligation.predicate.ty` (if we can).
pub fn project_and_unify_type<'cx,'tcx>(
    selcx: &mut SelectionContext<'cx,'tcx>,
    obligation: &ProjectionObligation<'tcx>)
    -> ProjectionResult<'tcx, ()>
{
    debug!("project_and_unify(obligation={})",
           obligation.repr(selcx.tcx()));

    let ty_obligation = obligation.with(obligation.predicate.projection_ty.clone());
    let projected_ty = try!(project_type(selcx, &ty_obligation));
    let infcx = selcx.infcx();
    let origin = infer::RelateOutputImplTypes(obligation.cause.span);
    debug!("project_and_unify_type: projected_ty = {}", projected_ty.repr(selcx.tcx()));
    match infer::mk_eqty(infcx, true, origin, projected_ty, obligation.predicate.ty) {
        Ok(()) => Ok(()),
        Err(e) => Err(ProjectionError::MismatchedTypes(MismatchedProjectionTypes{err: e})),
    }
}

/// Compute the result of a projection type (if we can).
pub fn project_type<'cx,'tcx>(
    selcx: &mut SelectionContext<'cx,'tcx>,
    obligation: &ProjectionTyObligation<'tcx>)
    -> ProjectionResult<'tcx, Ty<'tcx>>
{
    debug!("project(obligation={})",
           obligation.repr(selcx.tcx()));

    let mut candidates = ProjectionTyCandidateSet {
        vec: Vec::new(),
        ambiguous: false,
    };

    let () = assemble_candidates_from_param_env(selcx,
                                                obligation,
                                                &mut candidates);

    let () = assemble_candidates_from_object_type(selcx,
                                                  obligation,
                                                  &mut candidates);

    if candidates.vec.is_empty() {
        // FIXME(#20297) -- In `select.rs` there is similar logic that
        // gives precedence to where-clauses, but it's a bit more
        // fine-grained. I was lazy here and just always give
        // precedence to where-clauses or other such sources over
        // actually dredging through impls. This logic probably should
        // be tightened up.

        let () = try!(assemble_candidates_from_impls(selcx,
                                                     obligation,
                                                     &mut candidates));
    }

    debug!("{} candidates, ambiguous={}",
           candidates.vec.len(),
           candidates.ambiguous);

    // We probably need some winnowing logic similar to select here.

    if candidates.ambiguous || candidates.vec.len() > 1 {
        return Err(ProjectionError::TooManyCandidates);
    }

    match candidates.vec.pop() {
        Some(candidate) => {
            Ok(try!(confirm_candidate(selcx, obligation, candidate)))
        }
        None => {
            Err(ProjectionError::NoCandidate)
        }
    }
}

/// The first thing we have to do is scan through the parameter
/// environment to see whether there are any projection predicates
/// there that can answer this question.
fn assemble_candidates_from_param_env<'cx,'tcx>(
    selcx: &mut SelectionContext<'cx,'tcx>,
    obligation:  &ProjectionTyObligation<'tcx>,
    candidate_set: &mut ProjectionTyCandidateSet<'tcx>)
{
    let env_predicates = selcx.param_env().caller_bounds.predicates.clone();
    let env_predicates = env_predicates.iter().cloned().collect();
    assemble_candidates_from_predicates(selcx, obligation, candidate_set, env_predicates);
}

fn assemble_candidates_from_predicates<'cx,'tcx>(
    selcx: &mut SelectionContext<'cx,'tcx>,
    obligation:  &ProjectionTyObligation<'tcx>,
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
                        obligation.predicate.trait_ref.to_poly_trait_ref();
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
    obligation:  &ProjectionTyObligation<'tcx>,
    candidate_set: &mut ProjectionTyCandidateSet<'tcx>)
{
    let infcx = selcx.infcx();
    let trait_ref = infcx.resolve_type_vars_if_possible(&obligation.predicate.trait_ref);
    debug!("assemble_candidates_from_object_type(trait_ref={})",
           trait_ref.repr(infcx.tcx));
    let self_ty = trait_ref.self_ty();
    let data = match self_ty.sty {
        ty::ty_trait(ref data) => data,
        _ => { return; }
    };
    let projection_bounds = data.projection_bounds_with_self_ty(selcx.tcx(), self_ty);
    let env_predicates = projection_bounds.iter()
                                          .map(|p| p.as_predicate())
                                          .collect();
    assemble_candidates_from_predicates(selcx, obligation, candidate_set, env_predicates)
}

fn assemble_candidates_from_impls<'cx,'tcx>(
    selcx: &mut SelectionContext<'cx,'tcx>,
    obligation:  &ProjectionTyObligation<'tcx>,
    candidate_set: &mut ProjectionTyCandidateSet<'tcx>)
    -> ProjectionResult<'tcx, ()>
{
    // If we are resolving `<T as TraitRef<...>>::Item == Type`,
    // start out by selecting the predicate `T as TraitRef<...>`:
    let trait_ref =
        obligation.predicate.trait_ref.to_poly_trait_ref();
    let trait_obligation =
        obligation.with(trait_ref.to_poly_trait_predicate());
    let vtable = match selcx.select(&trait_obligation) {
        Ok(Some(vtable)) => vtable,
        Ok(None) => {
            candidate_set.ambiguous = true;
            return Ok(());
        }
        Err(e) => {
            debug!("assemble_candidates_from_impls: selection error {}",
                   e.repr(selcx.tcx()));
            return Err(ProjectionError::TraitSelectionError(e));
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
    obligation:  &ProjectionTyObligation<'tcx>,
    candidate: ProjectionTyCandidate<'tcx>)
    -> ProjectionResult<'tcx, Ty<'tcx>>
{
    let infcx = selcx.infcx();

    debug!("confirm_candidate(candidate={}, obligation={})",
           candidate.repr(infcx.tcx),
           obligation.repr(infcx.tcx));

    let projected_ty = match candidate {
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

            projection.ty
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
                Some(ty) => ty,
                None => {
                    selcx.tcx().sess.span_bug(
                        obligation.cause.span,
                        format!("impl `{}` did not contain projection for `{}`",
                                impl_vtable.repr(selcx.tcx()),
                                obligation.repr(selcx.tcx())).as_slice());
                }
            }
        }
    };

    Ok(projected_ty)
}

impl<'tcx> Repr<'tcx> for ProjectionError<'tcx> {
    fn repr(&self, tcx: &ty::ctxt<'tcx>) -> String {
        match *self {
            ProjectionError::NoCandidate =>
                format!("NoCandidate"),
            ProjectionError::TooManyCandidates =>
                format!("NoCandidate"),
            ProjectionError::MismatchedTypes(ref m) =>
                format!("MismatchedTypes({})", m.repr(tcx)),
            ProjectionError::TraitSelectionError(ref e) =>
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

