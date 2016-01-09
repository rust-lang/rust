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
    Obligation,
    ObligationCauseCode,
    OutputTypeParameterMismatch,
    TraitNotObjectSafe,
    PredicateObligation,
    SelectionError,
    ObjectSafetyViolation,
    MethodViolationCode,
    object_safety_violations,
};

use fmt_macros::{Parser, Piece, Position};
use middle::def_id::DefId;
use middle::infer::InferCtxt;
use middle::ty::{self, ToPredicate, ToPolyTraitRef, TraitRef, Ty, TypeFoldable};
use middle::ty::fast_reject;
use util::nodemap::{FnvHashMap, FnvHashSet};

use std::cmp;
use std::fmt;
use syntax::attr::{AttributeMethods, AttrMetaMethods};
use syntax::codemap::Span;
use syntax::errors::DiagnosticBuilder;

#[derive(Debug, PartialEq, Eq, Hash)]
pub struct TraitErrorKey<'tcx> {
    span: Span,
    predicate: ty::Predicate<'tcx>
}

impl<'tcx> TraitErrorKey<'tcx> {
    fn from_error<'a>(infcx: &InferCtxt<'a, 'tcx>,
                      e: &FulfillmentError<'tcx>) -> Self {
        let predicate =
            infcx.resolve_type_vars_if_possible(&e.obligation.predicate);
        TraitErrorKey {
            span: e.obligation.cause.span,
            predicate: infcx.tcx.erase_regions(&predicate)
        }
    }
}

pub fn report_fulfillment_errors<'a, 'tcx>(infcx: &InferCtxt<'a, 'tcx>,
                                           errors: &Vec<FulfillmentError<'tcx>>) {
    for error in errors {
        report_fulfillment_error(infcx, error);
    }
}

fn report_fulfillment_error<'a, 'tcx>(infcx: &InferCtxt<'a, 'tcx>,
                                      error: &FulfillmentError<'tcx>) {
    let error_key = TraitErrorKey::from_error(infcx, error);
    debug!("report_fulfillment_errors({:?}) - key={:?}",
           error, error_key);
    if !infcx.reported_trait_errors.borrow_mut().insert(error_key) {
        debug!("report_fulfillment_errors: skipping duplicate");
        return;
    }
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

    // The TyError created by normalize_to_error can end up being unified
    // into all obligations: for example, if our obligation is something
    // like `$X = <() as Foo<$X>>::Out` and () does not implement Foo<_>,
    // then $X will be unified with TyError, but the error still needs to be
    // reported.
    if !infcx.tcx.sess.has_errors() || !predicate.references_error() {
        let mut err = struct_span_err!(infcx.tcx.sess, obligation.cause.span, E0271,
            "type mismatch resolving `{}`: {}",
            predicate,
            error.err);
        note_obligation_cause(infcx, &mut err, obligation);
        err.emit();
    }
}

fn report_on_unimplemented<'a, 'tcx>(infcx: &InferCtxt<'a, 'tcx>,
                                     trait_ref: &TraitRef<'tcx>,
                                     span: Span) -> Option<String> {
    let def_id = trait_ref.def_id;
    let mut report = None;
    for item in infcx.tcx.get_attrs(def_id).iter() {
        if item.check_name("rustc_on_unimplemented") {
            let err_sp = item.meta().span.substitute_dummy(span);
            let def = infcx.tcx.lookup_trait_def(def_id);
            let trait_str = def.trait_ref.to_string();
            if let Some(ref istring) = item.value_str() {
                let mut generic_map = def.generics.types.iter_enumerated()
                                         .map(|(param, i, gen)| {
                                               (gen.name.as_str().to_string(),
                                                trait_ref.substs.types.get(param, i)
                                                         .to_string())
                                              }).collect::<FnvHashMap<String, String>>();
                generic_map.insert("Self".to_string(),
                                   trait_ref.self_ty().to_string());
                let parser = Parser::new(&istring);
                let mut errored = false;
                let err: String = parser.filter_map(|p| {
                    match p {
                        Piece::String(s) => Some(s),
                        Piece::NextArgument(a) => match a.position {
                            Position::ArgumentNamed(s) => match generic_map.get(s) {
                                Some(val) => Some(val),
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

/// Reports that an overflow has occurred and halts compilation. We
/// halt compilation unconditionally because it is important that
/// overflows never be masked -- they basically represent computations
/// whose result could not be truly determined and thus we can't say
/// if the program type checks or not -- and they are unusual
/// occurrences in any case.
pub fn report_overflow_error<'a, 'tcx, T>(infcx: &InferCtxt<'a, 'tcx>,
                                          obligation: &Obligation<'tcx, T>,
                                          suggest_increasing_limit: bool)
                                          -> !
    where T: fmt::Display + TypeFoldable<'tcx>
{
    let predicate =
        infcx.resolve_type_vars_if_possible(&obligation.predicate);
    let mut err = struct_span_err!(infcx.tcx.sess, obligation.cause.span, E0275,
                                   "overflow evaluating the requirement `{}`",
                                   predicate);

    if suggest_increasing_limit {
        suggest_new_overflow_limit(infcx.tcx, &mut err, obligation.cause.span);
    }

    note_obligation_cause(infcx, &mut err, obligation);

    err.emit();
    infcx.tcx.sess.abort_if_errors();
    unreachable!();
}

/// Reports that a cycle was detected which led to overflow and halts
/// compilation. This is equivalent to `report_overflow_error` except
/// that we can give a more helpful error message (and, in particular,
/// we do not suggest increasing the overflow limit, which is not
/// going to help).
pub fn report_overflow_error_cycle<'a, 'tcx>(infcx: &InferCtxt<'a, 'tcx>,
                                             cycle: &Vec<PredicateObligation<'tcx>>)
                                             -> !
{
    assert!(cycle.len() > 1);

    debug!("report_overflow_error_cycle(cycle length = {})", cycle.len());

    let cycle = infcx.resolve_type_vars_if_possible(cycle);

    debug!("report_overflow_error_cycle: cycle={:?}", cycle);

    assert_eq!(&cycle[0].predicate, &cycle.last().unwrap().predicate);

    try_report_overflow_error_type_of_infinite_size(infcx, &cycle);
    report_overflow_error(infcx, &cycle[0], false);
}

/// If a cycle results from evaluated whether something is Sized, that
/// is a particular special case that always results from a struct or
/// enum definition that lacks indirection (e.g., `struct Foo { x: Foo
/// }`). We wish to report a targeted error for this case.
pub fn try_report_overflow_error_type_of_infinite_size<'a, 'tcx>(
    infcx: &InferCtxt<'a, 'tcx>,
    cycle: &[PredicateObligation<'tcx>])
{
    let sized_trait = match infcx.tcx.lang_items.sized_trait() {
        Some(v) => v,
        None => return,
    };
    let top_is_sized = {
        match cycle[0].predicate {
            ty::Predicate::Trait(ref data) => data.def_id() == sized_trait,
            _ => false,
        }
    };
    if !top_is_sized {
        return;
    }

    // The only way to have a type of infinite size is to have,
    // somewhere, a struct/enum type involved. Identify all such types
    // and report the cycle to the user.

    let struct_enum_tys: Vec<_> =
        cycle.iter()
             .flat_map(|obligation| match obligation.predicate {
                 ty::Predicate::Trait(ref data) => {
                     assert_eq!(data.def_id(), sized_trait);
                     let self_ty = data.skip_binder().trait_ref.self_ty(); // (*)
                     // (*) ok to skip binder because this is just
                     // error reporting and regions don't really
                     // matter
                     match self_ty.sty {
                         ty::TyEnum(..) | ty::TyStruct(..) => Some(self_ty),
                         _ => None,
                     }
                 }
                 _ => {
                     infcx.tcx.sess.span_bug(obligation.cause.span,
                                             &format!("Sized cycle involving non-trait-ref: {:?}",
                                                      obligation.predicate));
                 }
             })
             .collect();

    assert!(!struct_enum_tys.is_empty());

    // This is a bit tricky. We want to pick a "main type" in the
    // listing that is local to the current crate, so we can give a
    // good span to the user. But it might not be the first one in our
    // cycle list. So find the first one that is local and then
    // rotate.
    let (main_index, main_def_id) =
        struct_enum_tys.iter()
                       .enumerate()
                       .filter_map(|(index, ty)| match ty.sty {
                           ty::TyEnum(adt_def, _) | ty::TyStruct(adt_def, _) if adt_def.did.is_local() =>
                               Some((index, adt_def.did)),
                           _ =>
                               None,
                       })
                       .next()
                       .unwrap(); // should always be SOME local type involved!

    // Rotate so that the "main" type is at index 0.
    let struct_enum_tys: Vec<_> =
        struct_enum_tys.iter()
                       .cloned()
                       .skip(main_index)
                       .chain(struct_enum_tys.iter().cloned().take(main_index))
                       .collect();

    let tcx = infcx.tcx;
    let mut err = recursive_type_with_infinite_size_error(tcx, main_def_id);
    let len = struct_enum_tys.len();
    if len > 2 {
        let span = tcx.map.span_if_local(main_def_id).unwrap();
        err.fileline_note(span,
                          &format!("type `{}` is embedded within `{}`...",
                                   struct_enum_tys[0],
                                   struct_enum_tys[1]));
        for &next_ty in &struct_enum_tys[1..len-1] {
            err.fileline_note(span,
                              &format!("...which in turn is embedded within `{}`...", next_ty));
        }
        err.fileline_note(span,
                          &format!("...which in turn is embedded within `{}`, \
                                    completing the cycle.",
                                   struct_enum_tys[len-1]));
    }
    err.emit();
    infcx.tcx.sess.abort_if_errors();
    unreachable!();
}

pub fn recursive_type_with_infinite_size_error<'tcx>(tcx: &ty::ctxt<'tcx>,
                                                     type_def_id: DefId)
                                                     -> DiagnosticBuilder<'tcx>
{
    assert!(type_def_id.is_local());
    let span = tcx.map.span_if_local(type_def_id).unwrap();
    let mut err = struct_span_err!(tcx.sess, span, E0072, "recursive type `{}` has infinite size",
                                   tcx.item_path_str(type_def_id));
    err.fileline_help(span, &format!("insert indirection (e.g., a `Box`, `Rc`, or `&`) \
                                      at some point to make `{}` representable",
                                     tcx.item_path_str(type_def_id)));
    err
}

pub fn report_selection_error<'a, 'tcx>(infcx: &InferCtxt<'a, 'tcx>,
                                        obligation: &PredicateObligation<'tcx>,
                                        error: &SelectionError<'tcx>)
{
    match *error {
        SelectionError::Unimplemented => {
            if let ObligationCauseCode::CompareImplMethodObligation = obligation.cause.code {
                span_err!(
                    infcx.tcx.sess, obligation.cause.span, E0276,
                    "the requirement `{}` appears on the impl \
                     method but not on the corresponding trait method",
                    obligation.predicate);
            } else {
                match obligation.predicate {
                    ty::Predicate::Trait(ref trait_predicate) => {
                        let trait_predicate =
                            infcx.resolve_type_vars_if_possible(trait_predicate);

                        if !infcx.tcx.sess.has_errors() || !trait_predicate.references_error() {
                            let trait_ref = trait_predicate.to_poly_trait_ref();
                            let mut err = struct_span_err!(
                                infcx.tcx.sess, obligation.cause.span, E0277,
                                "the trait `{}` is not implemented for the type `{}`",
                                trait_ref, trait_ref.self_ty());

                            // Check if it has a custom "#[rustc_on_unimplemented]"
                            // error message, report with that message if it does
                            let custom_note = report_on_unimplemented(infcx, &trait_ref.0,
                                                                      obligation.cause.span);
                            if let Some(s) = custom_note {
                                err.fileline_note(obligation.cause.span, &s);
                            } else {
                                let simp = fast_reject::simplify_type(infcx.tcx,
                                                                      trait_ref.self_ty(),
                                                                      true);
                                let mut impl_candidates = Vec::new();
                                let trait_def = infcx.tcx.lookup_trait_def(trait_ref.def_id());

                                match simp {
                                    Some(simp) => trait_def.for_each_impl(infcx.tcx, |def_id| {
                                        let imp = infcx.tcx.impl_trait_ref(def_id).unwrap();
                                        let imp_simp = fast_reject::simplify_type(infcx.tcx,
                                                                                  imp.self_ty(),
                                                                                  true);
                                        if let Some(imp_simp) = imp_simp {
                                            if simp != imp_simp {
                                                return;
                                            }
                                        }
                                        impl_candidates.push(imp);
                                    }),
                                    None => trait_def.for_each_impl(infcx.tcx, |def_id| {
                                        impl_candidates.push(
                                            infcx.tcx.impl_trait_ref(def_id).unwrap());
                                    })
                                };

                                if impl_candidates.len() > 0 {
                                    err.fileline_help(
                                        obligation.cause.span,
                                        &format!("the following implementations were found:"));

                                    let end = cmp::min(4, impl_candidates.len());
                                    for candidate in &impl_candidates[0..end] {
                                        err.fileline_help(obligation.cause.span,
                                                          &format!("  {:?}", candidate));
                                    }
                                    if impl_candidates.len() > 4 {
                                        err.fileline_help(obligation.cause.span,
                                                          &format!("and {} others",
                                                                   impl_candidates.len()-4));
                                    }
                                }
                            }
                            note_obligation_cause(infcx, &mut err, obligation);
                            err.emit();
                        }
                    },
                    ty::Predicate::Equate(ref predicate) => {
                        let predicate = infcx.resolve_type_vars_if_possible(predicate);
                        let err = infcx.equality_predicate(obligation.cause.span,
                                                           &predicate).err().unwrap();
                        let mut err = struct_span_err!(
                            infcx.tcx.sess, obligation.cause.span, E0278,
                            "the requirement `{}` is not satisfied (`{}`)",
                            predicate,
                            err);
                        note_obligation_cause(infcx, &mut err, obligation);
                        err.emit();
                    }

                    ty::Predicate::RegionOutlives(ref predicate) => {
                        let predicate = infcx.resolve_type_vars_if_possible(predicate);
                        let err = infcx.region_outlives_predicate(obligation.cause.span,
                                                                  &predicate).err().unwrap();
                        let mut err = struct_span_err!(
                            infcx.tcx.sess, obligation.cause.span, E0279,
                            "the requirement `{}` is not satisfied (`{}`)",
                            predicate,
                            err);
                        note_obligation_cause(infcx, &mut err, obligation);
                        err.emit();
                    }

                    ty::Predicate::Projection(..) | ty::Predicate::TypeOutlives(..) => {
                        let predicate =
                            infcx.resolve_type_vars_if_possible(&obligation.predicate);
                        let mut err = struct_span_err!(
                            infcx.tcx.sess, obligation.cause.span, E0280,
                            "the requirement `{}` is not satisfied",
                            predicate);
                        note_obligation_cause(infcx, &mut err, obligation);
                        err.emit();
                    }

                    ty::Predicate::ObjectSafe(trait_def_id) => {
                        let violations = object_safety_violations(
                            infcx.tcx, trait_def_id);
                        let mut err = report_object_safety_error(infcx.tcx,
                                                                 obligation.cause.span,
                                                                 trait_def_id,
                                                                 violations);
                        note_obligation_cause(infcx, &mut err, obligation);
                        err.emit();
                    }

                    ty::Predicate::WellFormed(ty) => {
                        // WF predicates cannot themselves make
                        // errors. They can only block due to
                        // ambiguity; otherwise, they always
                        // degenerate into other obligations
                        // (which may fail).
                        infcx.tcx.sess.span_bug(
                            obligation.cause.span,
                            &format!("WF predicate not satisfied for {:?}", ty));
                    }
                }
            }
        }

        OutputTypeParameterMismatch(ref expected_trait_ref, ref actual_trait_ref, ref e) => {
            let expected_trait_ref = infcx.resolve_type_vars_if_possible(&*expected_trait_ref);
            let actual_trait_ref = infcx.resolve_type_vars_if_possible(&*actual_trait_ref);
            if !actual_trait_ref.self_ty().references_error() {
                let mut err = struct_span_err!(
                    infcx.tcx.sess, obligation.cause.span, E0281,
                    "type mismatch: the type `{}` implements the trait `{}`, \
                     but the trait `{}` is required ({})",
                    expected_trait_ref.self_ty(),
                    expected_trait_ref,
                    actual_trait_ref,
                    e);
                note_obligation_cause(infcx, &mut err, obligation);
                err.emit();
            }
        }

        TraitNotObjectSafe(did) => {
            let violations = object_safety_violations(infcx.tcx, did);
            let mut err = report_object_safety_error(infcx.tcx, obligation.cause.span, did,
                                                     violations);
            note_obligation_cause(infcx, &mut err, obligation);
            err.emit();
        }
    }
}

pub fn report_object_safety_error<'tcx>(tcx: &ty::ctxt<'tcx>,
                                        span: Span,
                                        trait_def_id: DefId,
                                        violations: Vec<ObjectSafetyViolation>)
                                        -> DiagnosticBuilder<'tcx>
{
    let mut err = struct_span_err!(
        tcx.sess, span, E0038,
        "the trait `{}` cannot be made into an object",
        tcx.item_path_str(trait_def_id));

    let mut reported_violations = FnvHashSet();
    for violation in violations {
        if !reported_violations.insert(violation.clone()) {
            continue;
        }
        match violation {
            ObjectSafetyViolation::SizedSelf => {
                err.fileline_note(
                    span,
                    "the trait cannot require that `Self : Sized`");
            }

            ObjectSafetyViolation::SupertraitSelf => {
                err.fileline_note(
                    span,
                    "the trait cannot use `Self` as a type parameter \
                     in the supertrait listing");
            }

            ObjectSafetyViolation::Method(method,
                                          MethodViolationCode::StaticMethod) => {
                err.fileline_note(
                    span,
                    &format!("method `{}` has no receiver",
                             method.name));
            }

            ObjectSafetyViolation::Method(method,
                                          MethodViolationCode::ReferencesSelf) => {
                err.fileline_note(
                    span,
                    &format!("method `{}` references the `Self` type \
                              in its arguments or return type",
                             method.name));
            }

            ObjectSafetyViolation::Method(method,
                                          MethodViolationCode::Generic) => {
                err.fileline_note(
                    span,
                    &format!("method `{}` has generic type parameters",
                             method.name));
            }
        }
    }
    err
}

pub fn maybe_report_ambiguity<'a, 'tcx>(infcx: &InferCtxt<'a, 'tcx>,
                                        obligation: &PredicateObligation<'tcx>) {
    // Unable to successfully determine, probably means
    // insufficient type information, but could mean
    // ambiguous impls. The latter *ought* to be a
    // coherence violation, so we don't report it here.

    let predicate = infcx.resolve_type_vars_if_possible(&obligation.predicate);

    debug!("maybe_report_ambiguity(predicate={:?}, obligation={:?})",
           predicate,
           obligation);

    match predicate {
        ty::Predicate::Trait(ref data) => {
            let trait_ref = data.to_poly_trait_ref();
            let self_ty = trait_ref.self_ty();
            let all_types = &trait_ref.substs().types;
            if all_types.references_error() {
            } else {
                // Typically, this ambiguity should only happen if
                // there are unresolved type inference variables
                // (otherwise it would suggest a coherence
                // failure). But given #21974 that is not necessarily
                // the case -- we can have multiple where clauses that
                // are only distinguished by a region, which results
                // in an ambiguity even when all types are fully
                // known, since we don't dispatch based on region
                // relationships.

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
                        need_type_info(infcx, obligation.cause.span, self_ty);
                    } else {
                        let mut err = struct_span_err!(infcx.tcx.sess, obligation.cause.span, E0283,
                                                       "type annotations required: \
                                                        cannot resolve `{}`",
                                                       predicate);
                        note_obligation_cause(infcx, &mut err, obligation);
                        err.emit();
                    }
                }
            }
        }

        ty::Predicate::WellFormed(ty) => {
            // Same hacky approach as above to avoid deluging user
            // with error messages.
            if !ty.references_error() && !infcx.tcx.sess.has_errors() {
                need_type_info(infcx, obligation.cause.span, ty);
            }
        }

        _ => {
            if !infcx.tcx.sess.has_errors() {
                let mut err = struct_span_err!(infcx.tcx.sess, obligation.cause.span, E0284,
                                               "type annotations required: cannot resolve `{}`",
                                               predicate);
                note_obligation_cause(infcx, &mut err, obligation);
                err.emit();
            }
        }
    }
}

fn need_type_info<'a, 'tcx>(infcx: &InferCtxt<'a, 'tcx>,
                            span: Span,
                            ty: Ty<'tcx>)
{
    span_err!(infcx.tcx.sess, span, E0282,
              "unable to infer enough type information about `{}`; \
               type annotations or generic parameter binding required",
              ty);
}

fn note_obligation_cause<'a, 'tcx, T>(infcx: &InferCtxt<'a, 'tcx>,
                                      err: &mut DiagnosticBuilder,
                                      obligation: &Obligation<'tcx, T>)
    where T: fmt::Display
{
    note_obligation_cause_code(infcx,
                               err,
                               &obligation.predicate,
                               obligation.cause.span,
                               &obligation.cause.code);
}

fn note_obligation_cause_code<'a, 'tcx, T>(infcx: &InferCtxt<'a, 'tcx>,
                                           err: &mut DiagnosticBuilder,
                                           predicate: &T,
                                           cause_span: Span,
                                           cause_code: &ObligationCauseCode<'tcx>)
    where T: fmt::Display
{
    let tcx = infcx.tcx;
    match *cause_code {
        ObligationCauseCode::MiscObligation => { }
        ObligationCauseCode::SliceOrArrayElem => {
            err.fileline_note(
                cause_span,
                "slice and array elements must have `Sized` type");
        }
        ObligationCauseCode::ProjectionWf(data) => {
            err.fileline_note(
                cause_span,
                &format!("required so that the projection `{}` is well-formed",
                         data));
        }
        ObligationCauseCode::ReferenceOutlivesReferent(ref_ty) => {
            err.fileline_note(
                cause_span,
                &format!("required so that reference `{}` does not outlive its referent",
                         ref_ty));
        }
        ObligationCauseCode::ItemObligation(item_def_id) => {
            let item_name = tcx.item_path_str(item_def_id);
            err.fileline_note(
                cause_span,
                &format!("required by `{}`", item_name));
        }
        ObligationCauseCode::ObjectCastObligation(object_ty) => {
            err.fileline_note(
                cause_span,
                &format!(
                    "required for the cast to the object type `{}`",
                    infcx.ty_to_string(object_ty)));
        }
        ObligationCauseCode::RepeatVec => {
            err.fileline_note(
                cause_span,
                "the `Copy` trait is required because the \
                 repeated element will be copied");
        }
        ObligationCauseCode::VariableType(_) => {
            err.fileline_note(
                cause_span,
                "all local variables must have a statically known size");
        }
        ObligationCauseCode::ReturnType => {
            err.fileline_note(
                cause_span,
                "the return type of a function must have a \
                 statically known size");
        }
        ObligationCauseCode::AssignmentLhsSized => {
            err.fileline_note(
                cause_span,
                "the left-hand-side of an assignment must have a statically known size");
        }
        ObligationCauseCode::StructInitializerSized => {
            err.fileline_note(
                cause_span,
                "structs must have a statically known size to be initialized");
        }
        ObligationCauseCode::ClosureCapture(var_id, _, builtin_bound) => {
            let def_id = tcx.lang_items.from_builtin_kind(builtin_bound).unwrap();
            let trait_name = tcx.item_path_str(def_id);
            let name = tcx.local_var_name_str(var_id);
            err.fileline_note(
                cause_span,
                &format!("the closure that captures `{}` requires that all captured variables \
                          implement the trait `{}`",
                         name,
                         trait_name));
        }
        ObligationCauseCode::FieldSized => {
            err.fileline_note(
                cause_span,
                "only the last field of a struct or enum variant \
                 may have a dynamically sized type");
        }
        ObligationCauseCode::SharedStatic => {
            err.fileline_note(
                cause_span,
                "shared static variables must have a type that implements `Sync`");
        }
        ObligationCauseCode::BuiltinDerivedObligation(ref data) => {
            let parent_trait_ref = infcx.resolve_type_vars_if_possible(&data.parent_trait_ref);
            err.fileline_note(
                cause_span,
                &format!("required because it appears within the type `{}`",
                         parent_trait_ref.0.self_ty()));
            let parent_predicate = parent_trait_ref.to_predicate();
            note_obligation_cause_code(infcx,
                                       err,
                                       &parent_predicate,
                                       cause_span,
                                       &*data.parent_code);
        }
        ObligationCauseCode::ImplDerivedObligation(ref data) => {
            let parent_trait_ref = infcx.resolve_type_vars_if_possible(&data.parent_trait_ref);
            err.fileline_note(
                cause_span,
                &format!("required because of the requirements on the impl of `{}` for `{}`",
                         parent_trait_ref,
                         parent_trait_ref.0.self_ty()));
            let parent_predicate = parent_trait_ref.to_predicate();
            note_obligation_cause_code(infcx,
                                       err,
                                       &parent_predicate,
                                       cause_span,
                                       &*data.parent_code);
        }
        ObligationCauseCode::CompareImplMethodObligation => {
            err.fileline_note(
                cause_span,
                &format!("the requirement `{}` appears on the impl method \
                          but not on the corresponding trait method",
                         predicate));
        }
    }
}

fn suggest_new_overflow_limit(tcx: &ty::ctxt, err:&mut DiagnosticBuilder, span: Span) {
    let current_limit = tcx.sess.recursion_limit.get();
    let suggested_limit = current_limit * 2;
    err.fileline_note(
        span,
        &format!(
            "consider adding a `#![recursion_limit=\"{}\"]` attribute to your crate",
            suggested_limit));
}
