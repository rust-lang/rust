// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use super::{FulfillmentError, FulfillmentErrorCode,
            ObligationCauseCode, SelectionError,
            PredicateObligation, OutputTypeParameterMismatch};

use middle::infer::InferCtxt;
use middle::ty::{mod};
use syntax::codemap::Span;
use util::ppaux::{Repr, UserString};

pub fn report_fulfillment_errors<'a, 'tcx>(infcx: &InferCtxt<'a, 'tcx>,
                                           errors: &Vec<FulfillmentError<'tcx>>) {
    for error in errors.iter() {
        report_fulfillment_error(infcx, error);
    }
}

fn report_fulfillment_error<'a, 'tcx>(infcx: &InferCtxt<'a, 'tcx>,
                                      error: &FulfillmentError<'tcx>) {
    match error.code {
        FulfillmentErrorCode::CodeSelectionError(ref e) => {
            report_selection_error(infcx, &error.obligation, e);
        }
        FulfillmentErrorCode::CodeAmbiguity => {
            maybe_report_ambiguity(infcx, &error.obligation);
        }
    }
}

pub fn report_selection_error<'a, 'tcx>(infcx: &InferCtxt<'a, 'tcx>,
                                        obligation: &PredicateObligation<'tcx>,
                                        error: &SelectionError<'tcx>)
{
    match *error {
        SelectionError::Overflow => {
            // We could track the stack here more precisely if we wanted, I imagine.
            match obligation.trait_ref {
                ty::Predicate::Trait(ref trait_ref) => {
                    let trait_ref =
                        infcx.resolve_type_vars_if_possible(&**trait_ref);
                    infcx.tcx.sess.span_err(
                        obligation.cause.span,
                        format!(
                            "overflow evaluating the trait `{}` for the type `{}`",
                            trait_ref.user_string(infcx.tcx),
                            trait_ref.self_ty().user_string(infcx.tcx))[]);
                }

                ty::Predicate::Equate(ref predicate) => {
                    let predicate = infcx.resolve_type_vars_if_possible(predicate);
                    let err = infcx.equality_predicate(obligation.cause.span,
                                                       &predicate).unwrap_err();

                    infcx.tcx.sess.span_err(
                        obligation.cause.span,
                        format!(
                            "the requirement `{}` is not satisfied (`{}`)",
                            predicate.user_string(infcx.tcx),
                            ty::type_err_to_str(infcx.tcx, &err)).as_slice());
                }

                ty::Predicate::TypeOutlives(..) |
                ty::Predicate::RegionOutlives(..) => {
                    infcx.tcx.sess.span_err(
                        obligation.cause.span,
                        format!("overflow evaluating lifetime predicate").as_slice());
                }
            }

            let current_limit = infcx.tcx.sess.recursion_limit.get();
            let suggested_limit = current_limit * 2;
            infcx.tcx.sess.span_note(
                obligation.cause.span,
                format!(
                    "consider adding a `#![recursion_limit=\"{}\"]` attribute to your crate",
                    suggested_limit)[]);

            note_obligation_cause(infcx, obligation);
        }
        SelectionError::Unimplemented => {
            match obligation.trait_ref {
                ty::Predicate::Trait(ref trait_ref) => {
                    let trait_ref =
                        infcx.resolve_type_vars_if_possible(
                            &**trait_ref);
                    if !ty::type_is_error(trait_ref.self_ty()) {
                        infcx.tcx.sess.span_err(
                            obligation.cause.span,
                            format!(
                                "the trait `{}` is not implemented for the type `{}`",
                                trait_ref.user_string(infcx.tcx),
                                trait_ref.self_ty().user_string(infcx.tcx)).as_slice());
                        note_obligation_cause(infcx, obligation);
                    }
                }

                ty::Predicate::Equate(ref predicate) => {
                    let predicate = infcx.resolve_type_vars_if_possible(predicate);
                    let err = infcx.equality_predicate(obligation.cause.span,
                                                       &predicate).unwrap_err();

                    infcx.tcx.sess.span_err(
                        obligation.cause.span,
                        format!(
                            "the requirement `{}` is not satisfied (`{}`)",
                            predicate.user_string(infcx.tcx),
                            ty::type_err_to_str(infcx.tcx, &err)).as_slice());
                }

                ty::Predicate::TypeOutlives(..) |
                ty::Predicate::RegionOutlives(..) => {
                    let predicate = infcx.resolve_type_vars_if_possible(&obligation.trait_ref);
                    infcx.tcx.sess.span_err(
                        obligation.cause.span,
                        format!(
                            "the requirement `{}` is not satisfied",
                            predicate.user_string(infcx.tcx)).as_slice());
                }
            }
        }
        OutputTypeParameterMismatch(ref expected_trait_ref, ref actual_trait_ref, ref e) => {
            let expected_trait_ref =
                infcx.resolve_type_vars_if_possible(
                    &**expected_trait_ref);
            let actual_trait_ref =
                infcx.resolve_type_vars_if_possible(
                    &**actual_trait_ref);
            if !ty::type_is_error(actual_trait_ref.self_ty()) {
                infcx.tcx.sess.span_err(
                    obligation.cause.span,
                    format!(
                        "type mismatch: the type `{}` implements the trait `{}`, \
                         but the trait `{}` is required ({})",
                        expected_trait_ref.self_ty().user_string(infcx.tcx),
                        expected_trait_ref.user_string(infcx.tcx),
                        actual_trait_ref.user_string(infcx.tcx),
                        ty::type_err_to_str(infcx.tcx, e)).as_slice());
                note_obligation_cause(infcx, obligation);
            }
        }
    }
}

fn maybe_report_ambiguity<'a, 'tcx>(infcx: &InferCtxt<'a, 'tcx>,
                                    obligation: &PredicateObligation<'tcx>) {
    // Unable to successfully determine, probably means
    // insufficient type information, but could mean
    // ambiguous impls. The latter *ought* to be a
    // coherence violation, so we don't report it here.

    let trait_ref = match obligation.trait_ref {
        ty::Predicate::Trait(ref trait_ref) => {
            infcx.resolve_type_vars_if_possible(&**trait_ref)
        }
        _ => {
            infcx.tcx.sess.span_bug(
                obligation.cause.span,
                format!("ambiguity from something other than a trait: {}",
                        obligation.trait_ref.repr(infcx.tcx)).as_slice());
        }
    };
    let self_ty = trait_ref.self_ty();

    debug!("maybe_report_ambiguity(trait_ref={}, self_ty={}, obligation={})",
           trait_ref.repr(infcx.tcx),
           self_ty.repr(infcx.tcx),
           obligation.repr(infcx.tcx));
    let all_types = &trait_ref.substs().types;
    if all_types.iter().any(|&t| ty::type_is_error(t)) {
    } else if all_types.iter().any(|&t| ty::type_needs_infer(t)) {
        // This is kind of a hack: it frequently happens that some earlier
        // error prevents types from being fully inferred, and then we get
        // a bunch of uninteresting errors saying something like "<generic
        // #0> doesn't implement Sized".  It may even be true that we
        // could just skip over all checks where the self-ty is an
        // inference variable, but I was afraid that there might be an
        // inference variable created, registered as an obligation, and
        // then never forced by writeback, and hence by skipping here we'd
        // be ignoring the fact that we don't KNOW the type works
        // out. Though even that would probably be harmless, given that
        // we're only talking about builtin traits, which are known to be
        // inhabited. But in any case I just threw in this check for
        // has_errors() to be sure that compilation isn't happening
        // anyway. In that case, why inundate the user.
        if !infcx.tcx.sess.has_errors() {
            if infcx.tcx.lang_items.sized_trait()
                  .map_or(false, |sized_id| sized_id == trait_ref.def_id()) {
                infcx.tcx.sess.span_err(
                    obligation.cause.span,
                    format!(
                        "unable to infer enough type information about `{}`; type annotations \
                         required",
                        self_ty.user_string(infcx.tcx)).as_slice());
            } else {
                infcx.tcx.sess.span_err(
                    obligation.cause.span,
                    format!(
                        "unable to infer enough type information to \
                         locate the impl of the trait `{}` for \
                         the type `{}`; type annotations required",
                        trait_ref.user_string(infcx.tcx),
                        self_ty.user_string(infcx.tcx))[]);
                note_obligation_cause(infcx, obligation);
            }
        }
    } else if !infcx.tcx.sess.has_errors() {
         // Ambiguity. Coherence should have reported an error.
        infcx.tcx.sess.span_bug(
            obligation.cause.span,
            format!(
                "coherence failed to report ambiguity: \
                 cannot locate the impl of the trait `{}` for \
                 the type `{}`",
                trait_ref.user_string(infcx.tcx),
                self_ty.user_string(infcx.tcx))[]);
    }
}

fn note_obligation_cause<'a, 'tcx>(infcx: &InferCtxt<'a, 'tcx>,
                                   obligation: &PredicateObligation<'tcx>)
{
    let trait_ref = match obligation.trait_ref {
        ty::Predicate::Trait(ref trait_ref) => {
            infcx.resolve_type_vars_if_possible(&**trait_ref)
        }
        _ => {
            infcx.tcx.sess.span_bug(
                obligation.cause.span,
                format!("ambiguity from something other than a trait: {}",
                        obligation.trait_ref.repr(infcx.tcx)).as_slice());
        }
    };

    note_obligation_cause_code(infcx,
                               &trait_ref,
                               obligation.cause.span,
                               &obligation.cause.code)
}

fn note_obligation_cause_code<'a, 'tcx>(infcx: &InferCtxt<'a, 'tcx>,
                                        trait_ref: &ty::PolyTraitRef<'tcx>,
                                        cause_span: Span,
                                        cause_code: &ObligationCauseCode<'tcx>)
{
    let tcx = infcx.tcx;
    let trait_name = ty::item_path_str(tcx, trait_ref.def_id());
    match *cause_code {
        ObligationCauseCode::MiscObligation => { }
        ObligationCauseCode::ItemObligation(item_def_id) => {
            let item_name = ty::item_path_str(tcx, item_def_id);
            tcx.sess.span_note(
                cause_span,
                format!(
                    "the trait `{}` must be implemented because it is required by `{}`",
                    trait_name,
                    item_name).as_slice());
        }
        ObligationCauseCode::ObjectCastObligation(object_ty) => {
            tcx.sess.span_note(
                cause_span,
                format!(
                    "the trait `{}` must be implemented for the cast \
                     to the object type `{}`",
                    trait_name,
                    infcx.ty_to_string(object_ty)).as_slice());
        }
        ObligationCauseCode::RepeatVec => {
            tcx.sess.span_note(
                cause_span,
                "the `Copy` trait is required because the \
                 repeated element will be copied");
        }
        ObligationCauseCode::VariableType(_) => {
            tcx.sess.span_note(
                cause_span,
                "all local variables must have a statically known size");
        }
        ObligationCauseCode::ReturnType => {
            tcx.sess.span_note(
                cause_span,
                "the return type of a function must have a \
                 statically known size");
        }
        ObligationCauseCode::AssignmentLhsSized => {
            tcx.sess.span_note(
                cause_span,
                "the left-hand-side of an assignment must have a statically known size");
        }
        ObligationCauseCode::StructInitializerSized => {
            tcx.sess.span_note(
                cause_span,
                "structs must have a statically known size to be initialized");
        }
        ObligationCauseCode::ClosureCapture(var_id, closure_span, builtin_bound) => {
            let def_id = tcx.lang_items.from_builtin_kind(builtin_bound).unwrap();
            let trait_name = ty::item_path_str(tcx, def_id);
            let name = ty::local_var_name_str(tcx, var_id);
            span_note!(tcx.sess, closure_span,
                       "the closure that captures `{}` requires that all captured variables \
                       implement the trait `{}`",
                       name,
                       trait_name);
        }
        ObligationCauseCode::FieldSized => {
            span_note!(tcx.sess, cause_span,
                       "only the last field of a struct or enum variant \
                       may have a dynamically sized type")
        }
        ObligationCauseCode::ObjectSized => {
            span_note!(tcx.sess, cause_span,
                       "only sized types can be made into objects");
        }
        ObligationCauseCode::SharedStatic => {
            span_note!(tcx.sess, cause_span,
                       "shared static variables must have a type that implements `Sync`");
        }
        ObligationCauseCode::BuiltinDerivedObligation(ref root_trait_ref, ref root_cause_code) => {
            let root_trait_ref =
                infcx.resolve_type_vars_if_possible(&**root_trait_ref);
            span_note!(tcx.sess, cause_span,
                       "the type `{}` must implement `{}` because it appears within the type `{}`",
                       trait_ref.self_ty().user_string(infcx.tcx),
                       trait_ref.user_string(infcx.tcx),
                       root_trait_ref.self_ty().user_string(infcx.tcx));
            note_obligation_cause_code(infcx, &root_trait_ref, cause_span, &**root_cause_code);
        }
        ObligationCauseCode::ImplDerivedObligation(ref root_trait_ref, ref root_cause_code) => {
            let root_trait_ref =
                infcx.resolve_type_vars_if_possible(&**root_trait_ref);
            span_note!(tcx.sess, cause_span,
                       "the type `{}` must implement `{}` due to the requirements \
                        on the impl of `{}` for the type `{}`",
                       trait_ref.self_ty().user_string(infcx.tcx),
                       trait_ref.user_string(infcx.tcx),
                       root_trait_ref.user_string(infcx.tcx),
                       root_trait_ref.self_ty().user_string(infcx.tcx));
            note_obligation_cause_code(infcx, &root_trait_ref, cause_span, &**root_cause_code);
        }
    }
}
