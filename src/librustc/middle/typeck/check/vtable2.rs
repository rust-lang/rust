// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use middle::subst::{SelfSpace};
use middle::traits;
use middle::traits::{SelectionError, Overflow,
                     OutputTypeParameterMismatch, Unimplemented};
use middle::traits::{Obligation, obligation_for_builtin_bound};
use middle::traits::{FulfillmentError, Ambiguity};
use middle::traits::{ObligationCause};
use middle::ty;
use middle::typeck::check::{FnCtxt,
                            structurally_resolved_type};
use middle::typeck::infer;
use std::rc::Rc;
use syntax::ast;
use syntax::codemap::Span;
use util::ppaux::UserString;
use util::ppaux::Repr;

/// When reporting an error about a failed trait obligation, it's nice
/// to include some context indicating why we were checking that
/// obligation in the first place. The span is often enough but
/// sometimes it's not. Currently this enum is a bit of a hack and I
/// suspect it should be carried in the obligation or more deeply
/// integrated somehow.
pub enum ErrorReportingContext {
    GenericContext,
    ImplSupertraitCheck,
}

pub fn check_object_cast(fcx: &FnCtxt,
                         cast_expr: &ast::Expr,
                         source_expr: &ast::Expr,
                         target_object_ty: ty::t)
{
    debug!("check_object_cast(cast_expr={}, target_object_ty={})",
           cast_expr.repr(fcx.tcx()),
           target_object_ty.repr(fcx.tcx()));

    // Look up vtables for the type we're casting to,
    // passing in the source and target type.  The source
    // must be a pointer type suitable to the object sigil,
    // e.g.: `&x as &Trait` or `box x as Box<Trait>`
    let source_ty = fcx.expr_ty(source_expr);
    let source_ty = structurally_resolved_type(fcx, source_expr.span, source_ty);
    debug!("source_ty={}", source_ty.repr(fcx.tcx()));
    match (&ty::get(source_ty).sty, &ty::get(target_object_ty).sty) {
        (&ty::ty_uniq(referent_ty), &ty::ty_uniq(object_trait_ty)) => {
            let object_trait = object_trait(&object_trait_ty);

            // Ensure that if ~T is cast to ~Trait, then T : Trait
            push_cast_obligation(fcx, cast_expr, object_trait, referent_ty);
        }

        (&ty::ty_rptr(referent_region, ty::mt { ty: referent_ty,
                                                mutbl: referent_mutbl }),
         &ty::ty_rptr(target_region, ty::mt { ty: object_trait_ty,
                                              mutbl: target_mutbl })) =>
        {
            let object_trait = object_trait(&object_trait_ty);
            if !mutability_allowed(referent_mutbl, target_mutbl) {
                fcx.tcx().sess.span_err(source_expr.span,
                                        "types differ in mutability");
            } else {
                // Ensure that if &'a T is cast to &'b Trait, then T : Trait
                push_cast_obligation(fcx, cast_expr,
                                     object_trait,
                                     referent_ty);

                // Ensure that if &'a T is cast to &'b Trait, then 'b <= 'a
                infer::mk_subr(fcx.infcx(),
                               infer::RelateObjectBound(source_expr.span),
                               target_region,
                               referent_region);
            }
        }

        (_, &ty::ty_uniq(..)) => {
            fcx.ccx.tcx.sess.span_err(
                source_expr.span,
                format!("can only cast an boxed pointer \
                         to a boxed object, not a {}",
                        ty::ty_sort_string(fcx.tcx(), source_ty)).as_slice());
        }

        (_, &ty::ty_rptr(..)) => {
            fcx.ccx.tcx.sess.span_err(
                source_expr.span,
                format!("can only cast a &-pointer \
                         to an &-object, not a {}",
                        ty::ty_sort_string(fcx.tcx(), source_ty)).as_slice());
        }

        _ => {
            fcx.tcx().sess.span_bug(
                source_expr.span,
                "expected object type");
        }
    }

    // Because we currently give unsound lifetimes to the "ty_box", I
    // could have written &'static ty::TyTrait here, but it seems
    // gratuitously unsafe.
    fn object_trait<'a>(t: &'a ty::t) -> &'a ty::TyTrait {
        match ty::get(*t).sty {
            ty::ty_trait(ref ty_trait) => &**ty_trait,
            _ => fail!("expected ty_trait")
        }
    }

    fn mutability_allowed(a_mutbl: ast::Mutability,
                          b_mutbl: ast::Mutability)
                          -> bool {
        a_mutbl == b_mutbl ||
            (a_mutbl == ast::MutMutable && b_mutbl == ast::MutImmutable)
    }

    fn push_cast_obligation(fcx: &FnCtxt,
                            cast_expr: &ast::Expr,
                            object_trait: &ty::TyTrait,
                            referent_ty: ty::t) {
        let object_trait_ref =
            register_object_cast_obligations(fcx,
                                             cast_expr.span,
                                             object_trait,
                                             referent_ty);

        // Finally record the object_trait_ref for use during trans
        // (it would prob be better not to do this, but it's just kind
        // of a pain to have to reconstruct it).
        fcx.write_object_cast(cast_expr.id, object_trait_ref);
    }
}

pub fn register_object_cast_obligations(fcx: &FnCtxt,
                                        span: Span,
                                        object_trait: &ty::TyTrait,
                                        referent_ty: ty::t)
                                        -> Rc<ty::TraitRef>
{
    // This is just for better error reporting. Kinda goofy. The object type stuff
    // needs some refactoring so there is a more convenient type to pass around.
    let object_trait_ty =
        ty::mk_trait(fcx.tcx(),
                     object_trait.def_id,
                     object_trait.substs.clone(),
                     object_trait.bounds);

    debug!("register_object_cast_obligations: referent_ty={} object_trait_ty={}",
           referent_ty.repr(fcx.tcx()),
           object_trait_ty.repr(fcx.tcx()));

    // Take the type parameters from the object type, but set
    // the Self type (which is unknown, for the object type)
    // to be the type we are casting from.
    let mut object_substs = object_trait.substs.clone();
    assert!(object_substs.self_ty().is_none());
    object_substs.types.push(SelfSpace, referent_ty);

    // Create the obligation for casting from T to Trait.
    let object_trait_ref =
        Rc::new(ty::TraitRef { def_id: object_trait.def_id,
                               substs: object_substs });
    let object_obligation =
        Obligation::new(
            ObligationCause::new(span,
                                 traits::ObjectCastObligation(object_trait_ty)),
            object_trait_ref.clone());
    fcx.register_obligation(object_obligation);

    // Create additional obligations for all the various builtin
    // bounds attached to the object cast. (In other words, if the
    // object type is Foo+Send, this would create an obligation
    // for the Send check.)
    for builtin_bound in object_trait.bounds.builtin_bounds.iter() {
        fcx.register_obligation(
            obligation_for_builtin_bound(
                fcx.tcx(),
                ObligationCause::new(span,
                                     traits::ObjectCastObligation(object_trait_ty)),
                referent_ty,
                builtin_bound));
    }

    object_trait_ref
}

pub fn select_all_fcx_obligations_or_error(fcx: &FnCtxt) {
    debug!("select_all_fcx_obligations_or_error");

    let mut fulfillment_cx = fcx.inh.fulfillment_cx.borrow_mut();
    let r =
        fulfillment_cx.select_all_or_error(
            fcx.infcx(),
            &fcx.inh.param_env,
            &*fcx.inh.unboxed_closures.borrow());
    match r {
        Ok(()) => { }
        Err(errors) => { report_fulfillment_errors(fcx, &errors); }
    }
}

pub fn check_builtin_bound_obligations(fcx: &FnCtxt) {
    /*!
     * Hacky second pass to check builtin-bounds obligations *after*
     * writeback occurs.
     */

    match
        fcx.inh.fulfillment_cx.borrow()
                              .check_builtin_bound_obligations(fcx.infcx())
    {
        Ok(()) => { }
        Err(errors) => { report_fulfillment_errors(fcx, &errors); }
    }
}

fn resolve_trait_ref(fcx: &FnCtxt, obligation: &Obligation)
                     -> (ty::TraitRef, ty::t)
{
    let trait_ref =
        fcx.infcx().resolve_type_vars_in_trait_ref_if_possible(
            &*obligation.trait_ref);
    let self_ty =
        trait_ref.substs.self_ty().unwrap();
    (trait_ref, self_ty)
}

pub fn report_fulfillment_errors(fcx: &FnCtxt,
                                 errors: &Vec<FulfillmentError>) {
    for error in errors.iter() {
        report_fulfillment_error(fcx, error);
    }
}

pub fn report_fulfillment_error(fcx: &FnCtxt,
                                error: &FulfillmentError) {
    match error.code {
        SelectionError(ref e) => {
            report_selection_error(fcx, &error.obligation, e);
        }
        Ambiguity => {
            maybe_report_ambiguity(fcx, &error.obligation);
        }
    }
}

pub fn report_selection_error(fcx: &FnCtxt,
                              obligation: &Obligation,
                              error: &SelectionError) {
    match *error {
        Unimplemented => {
            let (trait_ref, self_ty) = resolve_trait_ref(fcx, obligation);
            if !ty::type_is_error(self_ty) {
                fcx.tcx().sess.span_err(
                    obligation.cause.span,
                    format!(
                        "the trait `{}` is not implemented for the type `{}`",
                        trait_ref.user_string(fcx.tcx()),
                        self_ty.user_string(fcx.tcx())).as_slice());
                note_obligation_cause(fcx, obligation);
            }
        }
        Overflow => {
            report_overflow(fcx, obligation);
        }
        OutputTypeParameterMismatch(ref expected_trait_ref, ref e) => {
            let expected_trait_ref =
                fcx.infcx().resolve_type_vars_in_trait_ref_if_possible(
                    &**expected_trait_ref);
            let (trait_ref, self_ty) = resolve_trait_ref(fcx, obligation);
            if !ty::type_is_error(self_ty) {
                fcx.tcx().sess.span_err(
                    obligation.cause.span,
                    format!(
                        "type mismatch: the type `{}` implements the trait `{}`, \
                         but the trait `{}` is required ({})",
                        self_ty.user_string(fcx.tcx()),
                        expected_trait_ref.user_string(fcx.tcx()),
                        trait_ref.user_string(fcx.tcx()),
                        ty::type_err_to_str(fcx.tcx(), e)).as_slice());
                note_obligation_cause(fcx, obligation);
            }
        }
    }
}

pub fn report_overflow(fcx: &FnCtxt, obligation: &Obligation) {
    let (trait_ref, self_ty) = resolve_trait_ref(fcx, obligation);
    if ty::type_is_error(self_ty) {
        fcx.tcx().sess.span_err(
            obligation.cause.span,
            format!(
                "could not locate an impl of the trait `{}` for \
                 the type `{}` due to overflow; possible cyclic \
                 dependency between impls",
                trait_ref.user_string(fcx.tcx()),
                self_ty.user_string(fcx.tcx())).as_slice());
        note_obligation_cause(fcx, obligation);
    }
}

pub fn maybe_report_ambiguity(fcx: &FnCtxt, obligation: &Obligation) {
    // Unable to successfully determine, probably means
    // insufficient type information, but could mean
    // ambiguous impls. The latter *ought* to be a
    // coherence violation, so we don't report it here.
    let (trait_ref, self_ty) = resolve_trait_ref(fcx, obligation);
    debug!("maybe_report_ambiguity(trait_ref={}, self_ty={}, obligation={})",
           trait_ref.repr(fcx.tcx()),
           self_ty.repr(fcx.tcx()),
           obligation.repr(fcx.tcx()));
    if ty::type_is_error(self_ty) {
    } else if ty::type_needs_infer(self_ty) {
        fcx.tcx().sess.span_err(
            obligation.cause.span,
            format!(
                "unable to infer enough type information to \
             locate the impl of the trait `{}` for \
             the type `{}`; type annotations required",
            trait_ref.user_string(fcx.tcx()),
            self_ty.user_string(fcx.tcx())).as_slice());
        note_obligation_cause(fcx, obligation);
    } else if fcx.tcx().sess.err_count() == 0 {
         // Ambiguity. Coherence should have reported an error.
        fcx.tcx().sess.span_bug(
            obligation.cause.span,
            format!(
                "coherence failed to report ambiguity: \
                 cannot locate the impl of the trait `{}` for \
                 the type `{}`",
                trait_ref.user_string(fcx.tcx()),
                self_ty.user_string(fcx.tcx())).as_slice());
    }
}

pub fn select_fcx_obligations_where_possible(fcx: &FnCtxt) {
    /*! Select as many obligations as we can at present. */

    match
        fcx.inh.fulfillment_cx
        .borrow_mut()
        .select_where_possible(fcx.infcx(),
                               &fcx.inh.param_env,
                               &*fcx.inh.unboxed_closures.borrow())
    {
        Ok(()) => { }
        Err(errors) => { report_fulfillment_errors(fcx, &errors); }
    }
}

fn note_obligation_cause(fcx: &FnCtxt,
                         obligation: &Obligation) {
    let tcx = fcx.tcx();
    let trait_name = ty::item_path_str(tcx, obligation.trait_ref.def_id);
    match obligation.cause.code {
        traits::MiscObligation => { }
        traits::ItemObligation(item_def_id) => {
            let item_name = ty::item_path_str(tcx, item_def_id);
            tcx.sess.span_note(
                obligation.cause.span,
                format!(
                    "the trait `{}` must be implemented because it is required by `{}`",
                    trait_name,
                    item_name).as_slice());
        }
        traits::ObjectCastObligation(object_ty) => {
            tcx.sess.span_note(
                obligation.cause.span,
                format!(
                    "the trait `{}` must be implemented for the cast \
                     to the object type `{}`",
                    trait_name,
                    fcx.infcx().ty_to_string(object_ty)).as_slice());
        }
        traits::RepeatVec => {
            tcx.sess.span_note(
                obligation.cause.span,
                format!(
                    "the `Copy` trait is required because the \
                     repeated element will be copied").as_slice());
        }
        traits::VariableType(_) => {
            tcx.sess.span_note(
                obligation.cause.span,
                "all local variables must have a statically known size");
        }
        traits::AssignmentLhsSized => {
            tcx.sess.span_note(
                obligation.cause.span,
                "the left-hand-side of an assignment must have a statically known size");
        }
        traits::StructInitializerSized => {
            tcx.sess.span_note(
                obligation.cause.span,
                "structs must have a statically known size to be initialized");
        }
    }
}
