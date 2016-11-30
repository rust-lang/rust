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

use super::{SelectionContext, Obligation, ObligationCause};

use hir::def_id::{DefId, LOCAL_CRATE};
use ty::{self, Ty, TyCtxt};

use infer::{InferCtxt, InferOk};

#[derive(Copy, Clone)]
struct InferIsLocal(bool);

/// If there are types that satisfy both impls, returns a suitably-freshened
/// `ImplHeader` with those types substituted
pub fn overlapping_impls<'cx, 'gcx, 'tcx>(infcx: &InferCtxt<'cx, 'gcx, 'tcx>,
                                          impl1_def_id: DefId,
                                          impl2_def_id: DefId)
                                          -> Option<ty::ImplHeader<'tcx>>
{
    debug!("impl_can_satisfy(\
           impl1_def_id={:?}, \
           impl2_def_id={:?})",
           impl1_def_id,
           impl2_def_id);

    let selcx = &mut SelectionContext::intercrate(infcx);
    overlap(selcx, impl1_def_id, impl2_def_id)
}

/// Can both impl `a` and impl `b` be satisfied by a common type (including
/// `where` clauses)? If so, returns an `ImplHeader` that unifies the two impls.
fn overlap<'cx, 'gcx, 'tcx>(selcx: &mut SelectionContext<'cx, 'gcx, 'tcx>,
                            a_def_id: DefId,
                            b_def_id: DefId)
                            -> Option<ty::ImplHeader<'tcx>>
{
    debug!("overlap(a_def_id={:?}, b_def_id={:?})",
           a_def_id,
           b_def_id);

    let a_impl_header = ty::ImplHeader::with_fresh_ty_vars(selcx, a_def_id);
    let b_impl_header = ty::ImplHeader::with_fresh_ty_vars(selcx, b_def_id);

    debug!("overlap: a_impl_header={:?}", a_impl_header);
    debug!("overlap: b_impl_header={:?}", b_impl_header);

    // Do `a` and `b` unify? If not, no overlap.
    match selcx.infcx().eq_impl_headers(true,
                                        &ObligationCause::dummy(),
                                        &a_impl_header,
                                        &b_impl_header) {
        Ok(InferOk { obligations, .. }) => {
            // FIXME(#32730) propagate obligations
            assert!(obligations.is_empty());
        }
        Err(_) => return None
    }

    debug!("overlap: unification check succeeded");

    // Are any of the obligations unsatisfiable? If so, no overlap.
    let infcx = selcx.infcx();
    let opt_failing_obligation =
        a_impl_header.predicates
                     .iter()
                     .chain(&b_impl_header.predicates)
                     .map(|p| infcx.resolve_type_vars_if_possible(p))
                     .map(|p| Obligation { cause: ObligationCause::dummy(),
                                           recursion_depth: 0,
                                           predicate: p })
                     .find(|o| !selcx.evaluate_obligation(o));

    if let Some(failing_obligation) = opt_failing_obligation {
        debug!("overlap: obligation unsatisfiable {:?}", failing_obligation);
        return None
    }

    Some(selcx.infcx().resolve_type_vars_if_possible(&a_impl_header))
}

pub fn trait_ref_is_knowable<'a, 'gcx, 'tcx>(tcx: TyCtxt<'a, 'gcx, 'tcx>,
                                             trait_ref: &ty::TraitRef<'tcx>) -> bool
{
    debug!("trait_ref_is_knowable(trait_ref={:?})", trait_ref);

    // if the orphan rules pass, that means that no ancestor crate can
    // impl this, so it's up to us.
    if orphan_check_trait_ref(tcx, trait_ref, InferIsLocal(false)).is_ok() {
        debug!("trait_ref_is_knowable: orphan check passed");
        return true;
    }

    // if the trait is not marked fundamental, then it's always possible that
    // an ancestor crate will impl this in the future, if they haven't
    // already
    if
        trait_ref.def_id.krate != LOCAL_CRATE &&
        !tcx.has_attr(trait_ref.def_id, "fundamental")
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
    orphan_check_trait_ref(tcx, trait_ref, InferIsLocal(true)).is_err()
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
pub fn orphan_check<'a, 'gcx, 'tcx>(tcx: TyCtxt<'a, 'gcx, 'tcx>,
                                    impl_def_id: DefId)
                                    -> Result<(), OrphanCheckErr<'tcx>>
{
    debug!("orphan_check({:?})", impl_def_id);

    // We only except this routine to be invoked on implementations
    // of a trait, not inherent implementations.
    let trait_ref = tcx.impl_trait_ref(impl_def_id).unwrap();
    debug!("orphan_check: trait_ref={:?}", trait_ref);

    // If the *trait* is local to the crate, ok.
    if trait_ref.def_id.is_local() {
        debug!("trait {:?} is local to current crate",
               trait_ref.def_id);
        return Ok(());
    }

    orphan_check_trait_ref(tcx, &trait_ref, InferIsLocal(false))
}

fn orphan_check_trait_ref<'tcx>(tcx: TyCtxt,
                                trait_ref: &ty::TraitRef<'tcx>,
                                infer_is_local: InferIsLocal)
                                -> Result<(), OrphanCheckErr<'tcx>>
{
    debug!("orphan_check_trait_ref(trait_ref={:?}, infer_is_local={})",
           trait_ref, infer_is_local.0);

    // First, create an ordered iterator over all the type parameters to the trait, with the self
    // type appearing first.
    // Find the first input type that either references a type parameter OR
    // some local type.
    for input_ty in trait_ref.input_types() {
        if ty_is_local(tcx, input_ty, infer_is_local) {
            debug!("orphan_check_trait_ref: ty_is_local `{:?}`", input_ty);

            // First local input type. Check that there are no
            // uncovered type parameters.
            let uncovered_tys = uncovered_tys(tcx, input_ty, infer_is_local);
            for uncovered_ty in uncovered_tys {
                if let Some(param) = uncovered_ty.walk().find(|t| is_type_parameter(t)) {
                    debug!("orphan_check_trait_ref: uncovered type `{:?}`", param);
                    return Err(OrphanCheckErr::UncoveredTy(param));
                }
            }

            // OK, found local type, all prior types upheld invariant.
            return Ok(());
        }

        // Otherwise, enforce invariant that there are no type
        // parameters reachable.
        if !infer_is_local.0 {
            if let Some(param) = input_ty.walk().find(|t| is_type_parameter(t)) {
                debug!("orphan_check_trait_ref: uncovered type `{:?}`", param);
                return Err(OrphanCheckErr::UncoveredTy(param));
            }
        }
    }

    // If we exit above loop, never found a local type.
    debug!("orphan_check_trait_ref: no local type");
    return Err(OrphanCheckErr::NoLocalInputType);
}

fn uncovered_tys<'tcx>(tcx: TyCtxt, ty: Ty<'tcx>, infer_is_local: InferIsLocal)
                       -> Vec<Ty<'tcx>> {
    if ty_is_local_constructor(tcx, ty, infer_is_local) {
        vec![]
    } else if fundamental_ty(tcx, ty) {
        ty.walk_shallow()
          .flat_map(|t| uncovered_tys(tcx, t, infer_is_local))
          .collect()
    } else {
        vec![ty]
    }
}

fn is_type_parameter(ty: Ty) -> bool {
    match ty.sty {
        // FIXME(#20590) straighten story about projection types
        ty::TyProjection(..) | ty::TyParam(..) => true,
        _ => false,
    }
}

fn ty_is_local(tcx: TyCtxt, ty: Ty, infer_is_local: InferIsLocal) -> bool {
    ty_is_local_constructor(tcx, ty, infer_is_local) ||
        fundamental_ty(tcx, ty) && ty.walk_shallow().any(|t| ty_is_local(tcx, t, infer_is_local))
}

fn fundamental_ty(tcx: TyCtxt, ty: Ty) -> bool {
    match ty.sty {
        ty::TyBox(..) | ty::TyRef(..) => true,
        ty::TyAdt(def, _) => def.is_fundamental(),
        ty::TyDynamic(ref data, ..) => {
            data.principal().map_or(false, |p| tcx.has_attr(p.def_id(), "fundamental"))
        }
        _ => false
    }
}

fn ty_is_local_constructor(tcx: TyCtxt, ty: Ty, infer_is_local: InferIsLocal)-> bool {
    debug!("ty_is_local_constructor({:?})", ty);

    match ty.sty {
        ty::TyBool |
        ty::TyChar |
        ty::TyInt(..) |
        ty::TyUint(..) |
        ty::TyFloat(..) |
        ty::TyStr |
        ty::TyFnDef(..) |
        ty::TyFnPtr(_) |
        ty::TyArray(..) |
        ty::TySlice(..) |
        ty::TyRawPtr(..) |
        ty::TyRef(..) |
        ty::TyNever |
        ty::TyTuple(..) |
        ty::TyParam(..) |
        ty::TyProjection(..) => {
            false
        }

        ty::TyInfer(..) => {
            infer_is_local.0
        }

        ty::TyAdt(def, _) => {
            def.did.is_local()
        }

        ty::TyBox(_) => { // Box<T>
            let krate = tcx.lang_items.owned_box().map(|d| d.krate);
            krate == Some(LOCAL_CRATE)
        }

        ty::TyDynamic(ref tt, ..) => {
            tt.principal().map_or(false, |p| p.def_id().is_local())
        }

        ty::TyError => {
            true
        }

        ty::TyClosure(..) | ty::TyAnon(..) => {
            bug!("ty_is_local invoked on unexpected type: {:?}", ty)
        }
    }
}
