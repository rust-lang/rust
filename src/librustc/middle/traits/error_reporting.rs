// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use super::{
    FulfillmentError,
    FulfillmentErrorCode,
    MismatchedProjectionTypes,
    ObligationCauseCode,
    OutputTypeParameterMismatch,
    PredicateObligation,
    SelectionError,
};

use fmt_macros::{Parser, Piece, Position};
use middle::infer::InferCtxt;
use middle::ty::{self, AsPredicate, ReferencesError, ToPolyTraitRef, TraitRef};
use std::collections::HashMap;
use syntax::codemap::{DUMMY_SP, Span};
use syntax::attr::{AttributeMethods, AttrMetaMethods};
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
        FulfillmentErrorCode::CodeProjectionError(ref e) => {
            report_projection_error(infcx, &error.obligation, e);
        }
        FulfillmentErrorCode::CodeAmbiguity => {
            maybe_report_ambiguity(infcx, &error.obligation);
        }
    }
}

pub fn report_projection_error<'a, 'tcx>(infcx: &InferCtxt<'a, 'tcx>,
                                         obligation: &PredicateObligation<'tcx>,
                                         error: &MismatchedProjectionTypes<'tcx>)
{
    let predicate =
        infcx.resolve_type_vars_if_possible(&obligation.predicate);
    if !predicate.references_error() {
        span_err!(infcx.tcx.sess, obligation.cause.span, E0271,
                "type mismatch resolving `{}`: {}",
                predicate.user_string(infcx.tcx),
                ty::type_err_to_str(infcx.tcx, &error.err));
        note_obligation_cause(infcx, obligation);
    }
}

fn report_on_unimplemented<'a, 'tcx>(infcx: &InferCtxt<'a, 'tcx>,
                                     trait_ref: &TraitRef<'tcx>,
                                     span: Span) -> Option<String> {
    let def_id = trait_ref.def_id;
    let mut report = None;
    for item in ty::get_attrs(infcx.tcx, def_id).iter() {
        if item.check_name("rustc_on_unimplemented") {
            let err_sp = if item.meta().span == DUMMY_SP {
                span
            } else {
                item.meta().span
            };
            let def = ty::lookup_trait_def(infcx.tcx, def_id);
            let trait_str = def.trait_ref.user_string(infcx.tcx);
            if let Some(ref istring) = item.value_str() {
                let mut generic_map = def.generics.types.iter_enumerated()
                                         .map(|(param, i, gen)| {
                                               (gen.name.as_str().to_string(),
                                                trait_ref.substs.types.get(param, i)
                                                         .user_string(infcx.tcx))
                                              }).collect::<HashMap<String, String>>();
                generic_map.insert("Self".to_string(),
                                   trait_ref.self_ty().user_string(infcx.tcx));
                let parser = Parser::new(istring.get());
                let mut errored = false;
                let err: String = parser.filter_map(|p| {
                    match p {
                        Piece::String(s) => Some(s),
                        Piece::NextArgument(a) => match a.position {
                            Position::ArgumentNamed(s) => match generic_map.get(s) {
                                Some(val) => Some(val.as_slice()),
                                None => {
                                    span_err!(infcx.tcx.sess, err_sp, E0272,
                                                   "the #[rustc_on_unimplemented] \
                                                            attribute on \
                                                            trait definition for {} refers to \
                                                            non-existent type parameter {}",
                                                           trait_str, s);
                                    errored = true;
                                    None
                                }
                            },
                            _ => {
                                     span_err!(infcx.tcx.sess, err_sp, E0273,
                                               "the #[rustc_on_unimplemented] \
                                                        attribute on \
                                                        trait definition for {} must have named \
                                                        format arguments, \
                                                        eg `#[rustc_on_unimplemented = \
                                                        \"foo {{T}}\"]`",
                                                       trait_str);
                                errored = true;
                                None
                            }
                        }
                    }
                }).collect();
                // Report only if the format string checks out
                if !errored {
                    report = Some(err);
                }
            } else {
                span_err!(infcx.tcx.sess, err_sp, E0274,
                                        "the #[rustc_on_unimplemented] attribute on \
                                                 trait definition for {} must have a value, \
                                                 eg `#[rustc_on_unimplemented = \"foo\"]`",
                                                 trait_str);
            }
            break;
        }
    }
    report
}

pub fn report_selection_error<'a, 'tcx>(infcx: &InferCtxt<'a, 'tcx>,
                                        obligation: &PredicateObligation<'tcx>,
                                        error: &SelectionError<'tcx>)
{
    match *error {
        SelectionError::Overflow => {
            // We could track the stack here more precisely if we wanted, I imagine.
            let predicate =
                infcx.resolve_type_vars_if_possible(&obligation.predicate);
            span_err!(infcx.tcx.sess, obligation.cause.span, E0275,
                    "overflow evaluating the requirement `{}`",
                    predicate.user_string(infcx.tcx));

            suggest_new_overflow_limit(infcx.tcx, obligation.cause.span);

            note_obligation_cause(infcx, obligation);
        }

        SelectionError::Unimplemented => {
            match &obligation.cause.code {
                &ObligationCauseCode::CompareImplMethodObligation => {
                    span_err!(infcx.tcx.sess, obligation.cause.span, E0276,
                            "the requirement `{}` appears on the impl \
                            method but not on the corresponding trait method",
                            obligation.predicate.user_string(infcx.tcx));;
                }
                _ => {
                    match obligation.predicate {
                        ty::Predicate::Trait(ref trait_predicate) => {
                            let trait_predicate =
                                infcx.resolve_type_vars_if_possible(trait_predicate);

                            if !trait_predicate.references_error() {
                                let trait_ref = trait_predicate.to_poly_trait_ref();
                                span_err!(infcx.tcx.sess, obligation.cause.span, E0277,
                                        "the trait `{}` is not implemented for the type `{}`",
                                        trait_ref.user_string(infcx.tcx),
                                        trait_ref.self_ty().user_string(infcx.tcx));
                                // Check if it has a custom "#[rustc_on_unimplemented]"
                                // error message, report with that message if it does
                                let custom_note = report_on_unimplemented(infcx, &*trait_ref.0,
                                                                          obligation.cause.span);
                                if let Some(s) = custom_note {
                                    infcx.tcx.sess.span_note(obligation.cause.span,
                                                             s.as_slice());
                                }
                            }
                        }

                        ty::Predicate::Equate(ref predicate) => {
                            let predicate = infcx.resolve_type_vars_if_possible(predicate);
                            let err = infcx.equality_predicate(obligation.cause.span,
                                                               &predicate).err().unwrap();
                            span_err!(infcx.tcx.sess, obligation.cause.span, E0278,
                                    "the requirement `{}` is not satisfied (`{}`)",
                                    predicate.user_string(infcx.tcx),
                                    ty::type_err_to_str(infcx.tcx, &err));
                        }

                        ty::Predicate::RegionOutlives(ref predicate) => {
                            let predicate = infcx.resolve_type_vars_if_possible(predicate);
                            let err = infcx.region_outlives_predicate(obligation.cause.span,
                                                                      &predicate).err().unwrap();
                            span_err!(infcx.tcx.sess, obligation.cause.span, E0279,
                                    "the requirement `{}` is not satisfied (`{}`)",
                                    predicate.user_string(infcx.tcx),
                                    ty::type_err_to_str(infcx.tcx, &err));
                        }

                        ty::Predicate::Projection(..) | ty::Predicate::TypeOutlives(..) => {
                                let predicate =
                                    infcx.resolve_type_vars_if_possible(&obligation.predicate);
                                span_err!(infcx.tcx.sess, obligation.cause.span, E0280,
                                        "the requirement `{}` is not satisfied",
                                        predicate.user_string(infcx.tcx));
                        }
                    }
                }
            }
        }

        OutputTypeParameterMismatch(ref expected_trait_ref, ref actual_trait_ref, ref e) => {
            let expected_trait_ref = infcx.resolve_type_vars_if_possible(&*expected_trait_ref);
            let actual_trait_ref = infcx.resolve_type_vars_if_possible(&*actual_trait_ref);
            if !ty::type_is_error(actual_trait_ref.self_ty()) {
                span_err!(infcx.tcx.sess, obligation.cause.span, E0281,
                        "type mismatch: the type `{}` implements the trait `{}`, \
                        but the trait `{}` is required ({})",
                        expected_trait_ref.self_ty().user_string(infcx.tcx),
                        expected_trait_ref.user_string(infcx.tcx),
                        actual_trait_ref.user_string(infcx.tcx),
                        ty::type_err_to_str(infcx.tcx, e));
                    note_obligation_cause(infcx, obligation);
            }
        }
    }
}

pub fn maybe_report_ambiguity<'a, 'tcx>(infcx: &InferCtxt<'a, 'tcx>,
                                        obligation: &PredicateObligation<'tcx>) {
    // Unable to successfully determine, probably means
    // insufficient type information, but could mean
    // ambiguous impls. The latter *ought* to be a
    // coherence violation, so we don't report it here.

    let predicate = infcx.resolve_type_vars_if_possible(&obligation.predicate);

    debug!("maybe_report_ambiguity(predicate={}, obligation={})",
           predicate.repr(infcx.tcx),
           obligation.repr(infcx.tcx));

    match predicate {
        ty::Predicate::Trait(ref data) => {
            let trait_ref = data.to_poly_trait_ref();
            let self_ty = trait_ref.self_ty();
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
                    if
                        infcx.tcx.lang_items.sized_trait()
                        .map_or(false, |sized_id| sized_id == trait_ref.def_id())
                    {
                        span_err!(infcx.tcx.sess, obligation.cause.span, E0282,
                                "unable to infer enough type information about `{}`; \
                                 type annotations required",
                                self_ty.user_string(infcx.tcx));
                    } else {
                        span_err!(infcx.tcx.sess, obligation.cause.span, E0283,
                                "type annotations required: cannot resolve `{}`",
                                predicate.user_string(infcx.tcx));;
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
                        self_ty.user_string(infcx.tcx)).as_slice());
            }
        }

        _ => {
            if !infcx.tcx.sess.has_errors() {
                span_err!(infcx.tcx.sess, obligation.cause.span, E0284,
                        "type annotations required: cannot resolve `{}`",
                        predicate.user_string(infcx.tcx));;
                note_obligation_cause(infcx, obligation);
            }
        }
    }
}

fn note_obligation_cause<'a, 'tcx>(infcx: &InferCtxt<'a, 'tcx>,
                                   obligation: &PredicateObligation<'tcx>)
{
    note_obligation_cause_code(infcx,
                               &obligation.predicate,
                               obligation.cause.span,
                               &obligation.cause.code);
}

fn note_obligation_cause_code<'a, 'tcx>(infcx: &InferCtxt<'a, 'tcx>,
                                        predicate: &ty::Predicate<'tcx>,
                                        cause_span: Span,
                                        cause_code: &ObligationCauseCode<'tcx>)
{
    let tcx = infcx.tcx;
    match *cause_code {
        ObligationCauseCode::MiscObligation => { }
        ObligationCauseCode::ItemObligation(item_def_id) => {
            let item_name = ty::item_path_str(tcx, item_def_id);
            tcx.sess.span_note(
                cause_span,
                format!("required by `{}`", item_name).as_slice());
        }
        ObligationCauseCode::ObjectCastObligation(object_ty) => {
            tcx.sess.span_note(
                cause_span,
                format!(
                    "required for the cast to the object type `{}`",
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
        ObligationCauseCode::BuiltinDerivedObligation(ref data) => {
            let parent_trait_ref = infcx.resolve_type_vars_if_possible(&data.parent_trait_ref);
            span_note!(tcx.sess, cause_span,
                       "required because it appears within the type `{}`",
                       parent_trait_ref.0.self_ty().user_string(infcx.tcx));
            let parent_predicate = parent_trait_ref.as_predicate();
            note_obligation_cause_code(infcx, &parent_predicate, cause_span, &*data.parent_code);
        }
        ObligationCauseCode::ImplDerivedObligation(ref data) => {
            let parent_trait_ref = infcx.resolve_type_vars_if_possible(&data.parent_trait_ref);
            span_note!(tcx.sess, cause_span,
                       "required because of the requirements on the impl of `{}` for `{}`",
                       parent_trait_ref.user_string(infcx.tcx),
                       parent_trait_ref.0.self_ty().user_string(infcx.tcx));
            let parent_predicate = parent_trait_ref.as_predicate();
            note_obligation_cause_code(infcx, &parent_predicate, cause_span, &*data.parent_code);
        }
        ObligationCauseCode::CompareImplMethodObligation => {
            span_note!(tcx.sess, cause_span,
                      "the requirement `{}` appears on the impl method\
                      but not on the corresponding trait method",
                      predicate.user_string(infcx.tcx));
        }
    }
}

pub fn suggest_new_overflow_limit(tcx: &ty::ctxt, span: Span) {
    let current_limit = tcx.sess.recursion_limit.get();
    let suggested_limit = current_limit * 2;
    tcx.sess.span_note(
        span,
        &format!(
            "consider adding a `#![recursion_limit=\"{}\"]` attribute to your crate",
            suggested_limit)[]);
}
