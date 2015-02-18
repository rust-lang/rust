// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! See `doc.rs` for high-level documentation

use super::Normalized;
use super::SelectionContext;
use super::{ObligationCause};
use super::PredicateObligation;
use super::project;
use super::util;

use middle::subst::{Subst, TypeSpace};
use middle::ty::{self, ToPolyTraitRef, Ty};
use middle::infer::{self, InferCtxt};
use std::collections::HashSet;
use std::rc::Rc;
use syntax::ast;
use syntax::codemap::DUMMY_SP;
use util::ppaux::Repr;

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
    let (a_trait_ref, a_obligations) = impl_trait_ref_and_oblig(selcx, a_def_id);
    let (b_trait_ref, b_obligations) = impl_trait_ref_and_oblig(selcx, b_def_id);

    // Does `a <: b` hold? If not, no overlap.
    if let Err(_) = infer::mk_sub_poly_trait_refs(selcx.infcx(),
                                                  true,
                                                  infer::Misc(DUMMY_SP),
                                                  a_trait_ref.to_poly_trait_ref(),
                                                  b_trait_ref.to_poly_trait_ref()) {
        return false;
    }

    // Are any of the obligations unsatisfiable? If so, no overlap.
    a_obligations.iter()
                 .chain(b_obligations.iter())
                 .all(|o| selcx.evaluate_obligation(o))
}

/// Instantiate fresh variables for all bound parameters of the impl
/// and return the impl trait ref with those variables substituted.
fn impl_trait_ref_and_oblig<'a,'tcx>(selcx: &mut SelectionContext<'a,'tcx>,
                                     impl_def_id: ast::DefId)
                                     -> (Rc<ty::TraitRef<'tcx>>,
                                         Vec<PredicateObligation<'tcx>>)
{
    let impl_substs =
        &util::fresh_substs_for_impl(selcx.infcx(), DUMMY_SP, impl_def_id);
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
    debug!("impl_is_local({})", impl_def_id.repr(tcx));

    // We only except this routine to be invoked on implementations
    // of a trait, not inherent implementations.
    let trait_ref = ty::impl_trait_ref(tcx, impl_def_id).unwrap();
    debug!("trait_ref={}", trait_ref.repr(tcx));

    // If the *trait* is local to the crate, ok.
    if trait_ref.def_id.krate == ast::LOCAL_CRATE {
        debug!("trait {} is local to current crate",
               trait_ref.def_id.repr(tcx));
        return Ok(());
    }

    // First, create an ordered iterator over all the type parameters to the trait, with the self
    // type appearing first.
    let input_tys = Some(trait_ref.self_ty());
    let input_tys = input_tys.iter().chain(trait_ref.substs.types.get_slice(TypeSpace).iter());
    let mut input_tys = input_tys;

    // Find the first input type that either references a type parameter OR
    // some local type.
    match input_tys.find(|&&input_ty| references_local_or_type_parameter(tcx, input_ty)) {
        Some(&input_ty) => {
            // Within this first type, check that all type parameters are covered by a local
            // type constructor. Note that if there is no local type constructor, then any
            // type parameter at all will be an error.
            let covered_params = type_parameters_covered_by_ty(tcx, input_ty);
            let all_params = type_parameters_reachable_from_ty(input_ty);
            for &param in all_params.difference(&covered_params) {
                return Err(OrphanCheckErr::UncoveredTy(param));
            }
        }
        None => {
            return Err(OrphanCheckErr::NoLocalInputType);
        }
    }

    return Ok(());
}

fn ty_is_local_constructor<'tcx>(tcx: &ty::ctxt<'tcx>, ty: Ty<'tcx>) -> bool {
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
        ty::ty_param(..) |
        ty::ty_projection(..) => {
            false
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
        ty::ty_infer(..) |
        ty::ty_open(..) |
        ty::ty_err => {
            tcx.sess.bug(
                &format!("ty_is_local invoked on unexpected type: {}",
                        ty.repr(tcx))[])
        }
    }
}

fn type_parameters_covered_by_ty<'tcx>(tcx: &ty::ctxt<'tcx>,
                                       ty: Ty<'tcx>)
                                       -> HashSet<Ty<'tcx>>
{
    if ty_is_local_constructor(tcx, ty) {
        type_parameters_reachable_from_ty(ty)
    } else {
        ty.walk_children().flat_map(|t| type_parameters_covered_by_ty(tcx, t).into_iter()).collect()
    }
}

/// All type parameters reachable from `ty`
fn type_parameters_reachable_from_ty<'tcx>(ty: Ty<'tcx>) -> HashSet<Ty<'tcx>> {
    ty.walk().filter(|&t| is_type_parameter(t)).collect()
}

fn references_local_or_type_parameter<'tcx>(tcx: &ty::ctxt<'tcx>, ty: Ty<'tcx>) -> bool {
    ty.walk().any(|ty| is_type_parameter(ty) || ty_is_local_constructor(tcx, ty))
}

fn is_type_parameter<'tcx>(ty: Ty<'tcx>) -> bool {
    match ty.sty {
        // FIXME(#20590) straighten story about projection types
        ty::ty_projection(..) | ty::ty_param(..) => true,
        _ => false,
    }
}
