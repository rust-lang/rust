// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use super::{CheckEnv, FnCtxt, FnCtxtTyper};
use middle::traits::{self, ObjectSafetyViolation, MethodViolationCode};
use middle::traits::{Obligation, ObligationCause};
use middle::traits::report_fulfillment_errors;
use middle::ty::{self, Ty, AsPredicate};
use syntax::codemap::Span;
use util::ppaux::{Repr, UserString};


// Check that a trait is 'object-safe'. This should be checked whenever a trait object
// is created (by casting or coercion, etc.). A trait is object-safe if all its
// methods are object-safe. A trait method is object-safe if it does not take
// self by value, has no type parameters and does not use the `Self` type, except
// in self position.
pub fn check_object_safety<'tcx>(tcx: &ty::ctxt<'tcx>,
                                 object_trait: &ty::TyTrait<'tcx>,
                                 span: Span)
{
    let trait_def_id = object_trait.principal_def_id();

    if traits::is_object_safe(tcx, trait_def_id) {
        return;
    }

    span_err!(tcx.sess, span, E0038,
              "cannot convert to a trait object because trait `{}` is not object-safe",
              ty::item_path_str(tcx, trait_def_id));

    let violations = traits::object_safety_violations(tcx, trait_def_id);
    for violation in violations {
        match violation {
            ObjectSafetyViolation::SizedSelf => {
                tcx.sess.span_note(
                    span,
                    "the trait cannot require that `Self : Sized`");
            }

            ObjectSafetyViolation::SupertraitSelf => {
                tcx.sess.span_note(
                    span,
                    "the trait cannot use `Self` as a type parameter \
                     in the supertrait listing");
            }

            ObjectSafetyViolation::Method(method, MethodViolationCode::StaticMethod) => {
                tcx.sess.span_note(
                    span,
                    &format!("method `{}` has no receiver",
                             method.name.user_string(tcx)));
            }

            ObjectSafetyViolation::Method(method, MethodViolationCode::ReferencesSelf) => {
                tcx.sess.span_note(
                    span,
                    &format!("method `{}` references the `Self` type \
                              in its arguments or return type",
                             method.name.user_string(tcx)));
            }

            ObjectSafetyViolation::Method(method, MethodViolationCode::Generic) => {
                tcx.sess.span_note(
                    span,
                    &format!("method `{}` has generic type parameters",
                             method.name.user_string(tcx)));
            }
        }
    }
}

pub fn register_object_cast_obligations<'a, 'tcx>(check_env: &mut CheckEnv<'tcx>,
                                                  fcx: &FnCtxt<'a, 'tcx>,
                                                  span: Span,
                                                  object_trait: &ty::TyTrait<'tcx>,
                                                  referent_ty: Ty<'tcx>)
                                                  -> ty::PolyTraitRef<'tcx>
{
    // We can only make objects from sized types.
    fcx.register_builtin_bound(
        check_env,
        referent_ty,
        ty::BoundSized,
        traits::ObligationCause::new(span, fcx.body_id, traits::ObjectSized));

    // This is just for better error reporting. Kinda goofy. The object type stuff
    // needs some refactoring so there is a more convenient type to pass around.
    let object_trait_ty =
        ty::mk_trait(fcx.tcx(),
                     object_trait.principal.clone(),
                     object_trait.bounds.clone());

    debug!("register_object_cast_obligations: referent_ty={} object_trait_ty={}",
           referent_ty.repr(fcx.tcx()),
           object_trait_ty.repr(fcx.tcx()));

    let cause = ObligationCause::new(span,
                                     fcx.body_id,
                                     traits::ObjectCastObligation(object_trait_ty));

    // Create the obligation for casting from T to Trait.
    let object_trait_ref =
        object_trait.principal_trait_ref_with_self_ty(fcx.tcx(), referent_ty);
    let object_obligation =
        Obligation::new(cause.clone(), object_trait_ref.as_predicate());
    fcx.register_predicate(check_env, object_obligation);

    // Create additional obligations for all the various builtin
    // bounds attached to the object cast. (In other words, if the
    // object type is Foo+Send, this would create an obligation
    // for the Send check.)
    for builtin_bound in &object_trait.bounds.builtin_bounds {
        fcx.register_builtin_bound(
            check_env,
            referent_ty,
            builtin_bound,
            cause.clone());
    }

    // Create obligations for the projection predicates.
    let projection_bounds =
        object_trait.projection_bounds_with_self_ty(fcx.tcx(), referent_ty);
    for projection_bound in &projection_bounds {
        let projection_obligation =
            Obligation::new(cause.clone(), projection_bound.as_predicate());
        fcx.register_predicate(check_env, projection_obligation);
    }

    object_trait_ref
}

pub fn select_all_fcx_obligations_and_apply_defaults<'a, 'ctx>(check_env: &mut CheckEnv<'ctx>,
                                                               fcx: &FnCtxt<'a, 'ctx>) {
    debug!("select_all_fcx_obligations_and_apply_defaults");

    select_fcx_obligations_where_possible(check_env, fcx);
    fcx.default_type_parameters(check_env);
    select_fcx_obligations_where_possible(check_env, fcx);
}

pub fn select_all_fcx_obligations_or_error<'a, 'ctx>(check_env: &mut CheckEnv<'ctx>,
                                                     fcx: &FnCtxt<'a, 'ctx>) {
    debug!("select_all_fcx_obligations_or_error");

    // upvar inference should have ensured that all deferred call
    // resolutions are handled by now.
    assert!(check_env.deferred_call_resolutions.is_empty());

    select_all_fcx_obligations_and_apply_defaults(check_env, fcx);
    let typer = FnCtxtTyper::new(&check_env.tt, fcx);
    let r = check_env.fulfillment_cx.select_all_or_error(fcx.infcx(), &typer);
    match r {
        Ok(()) => { }
        Err(errors) => { report_fulfillment_errors(fcx.infcx(), &errors); }
    }
}

/// Select as many obligations as we can at present.
pub fn select_fcx_obligations_where_possible<'a, 'ctx>(check_env: &mut CheckEnv<'ctx>,
                                                       fcx: &FnCtxt<'a, 'ctx>)
{
    let typer = FnCtxtTyper::new(&check_env.tt, fcx);
    match
        check_env.fulfillment_cx
        .select_where_possible(fcx.infcx(), &typer)
    {
        Ok(()) => { }
        Err(errors) => { report_fulfillment_errors(fcx.infcx(), &errors); }
    }
}

/// Try to select any fcx obligation that we haven't tried yet, in an effort to improve inference.
/// You could just call `select_fcx_obligations_where_possible` except that it leads to repeated
/// work.
pub fn select_new_fcx_obligations<'a, 'ctx>(check_env: &mut CheckEnv<'ctx>,
                                            fcx: &FnCtxt<'a, 'ctx>)
{
    let typer = FnCtxtTyper::new(&check_env.tt, fcx);
    match
        check_env.fulfillment_cx
        .select_new_obligations(fcx.infcx(), &typer)
    {
        Ok(()) => { }
        Err(errors) => { report_fulfillment_errors(fcx.infcx(), &errors); }
    }
}
