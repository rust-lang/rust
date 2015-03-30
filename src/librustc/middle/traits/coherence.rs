// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! See `README.md` for high-level documentation

use super::Normalized;
use super::SelectionContext;
use super::ObligationCause;
use super::PredicateObligation;
use super::project;
use super::util;

use middle::subst::{Subst, Substs, TypeSpace};
use middle::ty::{self, ToPolyTraitRef, Ty};
use middle::infer::{self, InferCtxt};
use std::rc::Rc;
use syntax::ast;
use syntax::codemap::{DUMMY_SP, Span};
use util::ppaux::Repr;

#[derive(Copy)]
struct ParamIsLocal(bool);

/// True if there exist types that satisfy both of the two given impls.
pub fn overlapping_impls(infcx: &InferCtxt,
                         impl1_def_id: ast::DefId,
                         impl2_def_id: ast::DefId)
                         -> bool
{
    debug!("impl_can_satisfy(\
           impl1_def_id={}, \
           impl2_def_id={})",
           impl1_def_id.repr(infcx.tcx),
           impl2_def_id.repr(infcx.tcx));

    let param_env = &ty::empty_parameter_environment(infcx.tcx);
    let selcx = &mut SelectionContext::intercrate(infcx, param_env);
    infcx.probe(|_| {
        overlap(selcx, impl1_def_id, impl2_def_id) || overlap(selcx, impl2_def_id, impl1_def_id)
    })
}

/// Can the types from impl `a` be used to satisfy impl `b`?
/// (Including all conditions)
fn overlap(selcx: &mut SelectionContext,
           a_def_id: ast::DefId,
           b_def_id: ast::DefId)
           -> bool
{
    debug!("overlap(a_def_id={}, b_def_id={})",
           a_def_id.repr(selcx.tcx()),
           b_def_id.repr(selcx.tcx()));

    let (a_trait_ref, a_obligations) = impl_trait_ref_and_oblig(selcx,
                                                                a_def_id,
                                                                util::free_substs_for_impl);

    let (b_trait_ref, b_obligations) = impl_trait_ref_and_oblig(selcx,
                                                                b_def_id,
                                                                util::fresh_type_vars_for_impl);

    debug!("overlap: a_trait_ref={}", a_trait_ref.repr(selcx.tcx()));

    debug!("overlap: b_trait_ref={}", b_trait_ref.repr(selcx.tcx()));

    // Does `a <: b` hold? If not, no overlap.
    if let Err(_) = infer::mk_sub_poly_trait_refs(selcx.infcx(),
                                                  true,
                                                  infer::Misc(DUMMY_SP),
                                                  a_trait_ref.to_poly_trait_ref(),
                                                  b_trait_ref.to_poly_trait_ref()) {
        return false;
    }

    debug!("overlap: subtraitref check succeeded");

    // Are any of the obligations unsatisfiable? If so, no overlap.
    let tcx = selcx.tcx();
    let infcx = selcx.infcx();
    let opt_failing_obligation =
        a_obligations.iter()
                     .chain(b_obligations.iter())
                     .map(|o| infcx.resolve_type_vars_if_possible(o))
                     .find(|o| !selcx.evaluate_obligation(o));

    if let Some(failing_obligation) = opt_failing_obligation {
        debug!("overlap: obligation unsatisfiable {}", failing_obligation.repr(tcx));
        return false
    }

    true
}

pub fn trait_ref_is_knowable<'tcx>(tcx: &ty::ctxt<'tcx>, trait_ref: &ty::TraitRef<'tcx>) -> bool
{
    debug!("trait_ref_is_knowable(trait_ref={})", trait_ref.repr(tcx));

    // if the orphan rules pass, that means that no ancestor crate can
    // impl this, so it's up to us.
    if orphan_check_trait_ref(tcx, trait_ref, ParamIsLocal(false)).is_ok() {
        debug!("trait_ref_is_knowable: orphan check passed");
        return true;
    }

    // if the trait is not marked fundamental, then it's always possible that
    // an ancestor crate will impl this in the future, if they haven't
    // already
    if
        trait_ref.def_id.krate != ast::LOCAL_CRATE &&
        !ty::has_attr(tcx, trait_ref.def_id, "fundamental")
    {
        debug!("trait_ref_is_knowable: trait is neither local nor fundamental");
        return false;
    }

    // find out when some downstream (or cousin) crate could impl this
    // trait-ref, presuming that all the parameters were instantiated
    // with downstream types. If not, then it could only be
    // implemented by an upstream crate, which means that the impl
    // must be visible to us, and -- since the trait is fundamental
    // -- we can test.
    orphan_check_trait_ref(tcx, trait_ref, ParamIsLocal(true)).is_err()
}

type SubstsFn = for<'a,'tcx> fn(infcx: &InferCtxt<'a, 'tcx>,
                                span: Span,
                                impl_def_id: ast::DefId)
                                -> Substs<'tcx>;

/// Instantiate fresh variables for all bound parameters of the impl
/// and return the impl trait ref with those variables substituted.
fn impl_trait_ref_and_oblig<'a,'tcx>(selcx: &mut SelectionContext<'a,'tcx>,
                                     impl_def_id: ast::DefId,
                                     substs_fn: SubstsFn)
                                     -> (Rc<ty::TraitRef<'tcx>>,
                                         Vec<PredicateObligation<'tcx>>)
{
    let impl_substs =
        &substs_fn(selcx.infcx(), DUMMY_SP, impl_def_id);
    let impl_trait_ref =
        ty::impl_trait_ref(selcx.tcx(), impl_def_id).unwrap();
    let impl_trait_ref =
        impl_trait_ref.subst(selcx.tcx(), impl_substs);
    let Normalized { value: impl_trait_ref, obligations: normalization_obligations1 } =
        project::normalize(selcx, ObligationCause::dummy(), &impl_trait_ref);

    let predicates = ty::lookup_predicates(selcx.tcx(), impl_def_id);
    let predicates = predicates.instantiate(selcx.tcx(), impl_substs);
    let Normalized { value: predicates, obligations: normalization_obligations2 } =
        project::normalize(selcx, ObligationCause::dummy(), &predicates);
    let impl_obligations =
        util::predicates_for_generics(selcx.tcx(), ObligationCause::dummy(), 0, &predicates);

    let impl_obligations: Vec<_> =
        impl_obligations.into_iter()
        .chain(normalization_obligations1.into_iter())
        .chain(normalization_obligations2.into_iter())
        .collect();

    (impl_trait_ref, impl_obligations)
}

pub enum OrphanCheckErr<'tcx> {
    NoLocalInputType,
    UncoveredTy(Ty<'tcx>),
}

/// Checks the coherence orphan rules. `impl_def_id` should be the
/// def-id of a trait impl. To pass, either the trait must be local, or else
/// two conditions must be satisfied:
///
/// 1. All type parameters in `Self` must be "covered" by some local type constructor.
/// 2. Some local type must appear in `Self`.
pub fn orphan_check<'tcx>(tcx: &ty::ctxt<'tcx>,
                          impl_def_id: ast::DefId)
                          -> Result<(), OrphanCheckErr<'tcx>>
{
    debug!("orphan_check({})", impl_def_id.repr(tcx));

    // We only except this routine to be invoked on implementations
    // of a trait, not inherent implementations.
    let trait_ref = ty::impl_trait_ref(tcx, impl_def_id).unwrap();
    debug!("orphan_check: trait_ref={}", trait_ref.repr(tcx));

    // If the *trait* is local to the crate, ok.
    if trait_ref.def_id.krate == ast::LOCAL_CRATE {
        debug!("trait {} is local to current crate",
               trait_ref.def_id.repr(tcx));
        return Ok(());
    }

    orphan_check_trait_ref(tcx, &trait_ref, ParamIsLocal(false))
}

fn orphan_check_trait_ref<'tcx>(tcx: &ty::ctxt<'tcx>,
                                trait_ref: &ty::TraitRef<'tcx>,
                                param_is_local: ParamIsLocal)
                                -> Result<(), OrphanCheckErr<'tcx>>
{
    debug!("orphan_check_trait_ref(trait_ref={}, param_is_local={})",
           trait_ref.repr(tcx), param_is_local.0);

    // First, create an ordered iterator over all the type parameters to the trait, with the self
    // type appearing first.
    let input_tys = Some(trait_ref.self_ty());
    let input_tys = input_tys.iter().chain(trait_ref.substs.types.get_slice(TypeSpace).iter());

    // Find the first input type that either references a type parameter OR
    // some local type.
    for input_ty in input_tys {
        if ty_is_local(tcx, input_ty, param_is_local) {
            debug!("orphan_check_trait_ref: ty_is_local `{}`", input_ty.repr(tcx));

            // First local input type. Check that there are no
            // uncovered type parameters.
            let uncovered_tys = uncovered_tys(tcx, input_ty, param_is_local);
            for uncovered_ty in uncovered_tys {
                if let Some(param) = uncovered_ty.walk().find(|t| is_type_parameter(t)) {
                    debug!("orphan_check_trait_ref: uncovered type `{}`", param.repr(tcx));
                    return Err(OrphanCheckErr::UncoveredTy(param));
                }
            }

            // OK, found local type, all prior types upheld invariant.
            return Ok(());
        }

        // Otherwise, enforce invariant that there are no type
        // parameters reachable.
        if !param_is_local.0 {
            if let Some(param) = input_ty.walk().find(|t| is_type_parameter(t)) {
                debug!("orphan_check_trait_ref: uncovered type `{}`", param.repr(tcx));
                return Err(OrphanCheckErr::UncoveredTy(param));
            }
        }
    }

    // If we exit above loop, never found a local type.
    debug!("orphan_check_trait_ref: no local type");
    return Err(OrphanCheckErr::NoLocalInputType);
}

fn uncovered_tys<'tcx>(tcx: &ty::ctxt<'tcx>,
                       ty: Ty<'tcx>,
                       param_is_local: ParamIsLocal)
                       -> Vec<Ty<'tcx>>
{
    if ty_is_local_constructor(tcx, ty, param_is_local) {
        vec![]
    } else if fundamental_ty(tcx, ty) {
        ty.walk_shallow()
          .flat_map(|t| uncovered_tys(tcx, t, param_is_local).into_iter())
          .collect()
    } else {
        vec![ty]
    }
}

fn is_type_parameter<'tcx>(ty: Ty<'tcx>) -> bool {
    match ty.sty {
        // FIXME(#20590) straighten story about projection types
        ty::ty_projection(..) | ty::ty_param(..) => true,
        _ => false,
    }
}

fn ty_is_local<'tcx>(tcx: &ty::ctxt<'tcx>, ty: Ty<'tcx>, param_is_local: ParamIsLocal) -> bool
{
    ty_is_local_constructor(tcx, ty, param_is_local) ||
        fundamental_ty(tcx, ty) && ty.walk_shallow().any(|t| ty_is_local(tcx, t, param_is_local))
}

fn fundamental_ty<'tcx>(tcx: &ty::ctxt<'tcx>, ty: Ty<'tcx>) -> bool
{
    match ty.sty {
        ty::ty_uniq(..) | ty::ty_rptr(..) =>
            true,
        ty::ty_enum(def_id, _) | ty::ty_struct(def_id, _) =>
            ty::has_attr(tcx, def_id, "fundamental"),
        ty::ty_trait(ref data) =>
            ty::has_attr(tcx, data.principal_def_id(), "fundamental"),
        _ =>
            false
    }
}

fn ty_is_local_constructor<'tcx>(tcx: &ty::ctxt<'tcx>,
                                 ty: Ty<'tcx>,
                                 param_is_local: ParamIsLocal)
                                 -> bool
{
    debug!("ty_is_local_constructor({})", ty.repr(tcx));

    match ty.sty {
        ty::ty_bool |
        ty::ty_char |
        ty::ty_int(..) |
        ty::ty_uint(..) |
        ty::ty_float(..) |
        ty::ty_str(..) |
        ty::ty_bare_fn(..) |
        ty::ty_vec(..) |
        ty::ty_ptr(..) |
        ty::ty_rptr(..) |
        ty::ty_tup(..) |
        ty::ty_infer(..) |
        ty::ty_projection(..) => {
            false
        }

        ty::ty_param(..) => {
            param_is_local.0
        }

        ty::ty_enum(def_id, _) |
        ty::ty_struct(def_id, _) => {
            def_id.krate == ast::LOCAL_CRATE
        }

        ty::ty_uniq(_) => { // treat ~T like Box<T>
            let krate = tcx.lang_items.owned_box().map(|d| d.krate);
            krate == Some(ast::LOCAL_CRATE)
        }

        ty::ty_trait(ref tt) => {
            tt.principal_def_id().krate == ast::LOCAL_CRATE
        }

        ty::ty_closure(..) |
        ty::ty_err => {
            tcx.sess.bug(
                &format!("ty_is_local invoked on unexpected type: {}",
                        ty.repr(tcx)))
        }
    }
}


