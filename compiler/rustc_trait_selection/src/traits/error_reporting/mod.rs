pub mod on_unimplemented;
pub mod suggestions;

use super::{
    EvaluationResult, FulfillmentError, FulfillmentErrorCode, MismatchedProjectionTypes,
    Obligation, ObligationCause, ObligationCauseCode, OnUnimplementedDirective,
    OnUnimplementedNote, OutputTypeParameterMismatch, Overflow, PredicateObligation,
    SelectionContext, SelectionError, TraitNotObjectSafe,
};

use crate::infer::error_reporting::{TyCategory, TypeAnnotationNeeded as ErrorCode};
use crate::infer::type_variable::{TypeVariableOrigin, TypeVariableOriginKind};
use crate::infer::{self, InferCtxt, TyCtxtInferExt};
use rustc_data_structures::fx::FxHashMap;
use rustc_errors::{pluralize, struct_span_err, Applicability, DiagnosticBuilder, ErrorReported};
use rustc_hir as hir;
use rustc_hir::def_id::DefId;
use rustc_hir::intravisit::Visitor;
use rustc_hir::GenericParam;
use rustc_hir::Item;
use rustc_hir::Node;
use rustc_middle::thir::abstract_const::NotConstEvaluatable;
use rustc_middle::ty::error::ExpectedFound;
use rustc_middle::ty::fold::TypeFolder;
use rustc_middle::ty::{
    self, fast_reject, AdtKind, SubtypePredicate, ToPolyTraitRef, ToPredicate, Ty, TyCtxt,
    TypeFoldable, WithConstness,
};
use rustc_session::DiagnosticMessageId;
use rustc_span::symbol::{kw, sym};
use rustc_span::{ExpnKind, MultiSpan, Span, DUMMY_SP};
use std::fmt;
use std::iter;

use crate::traits::query::evaluate_obligation::InferCtxtExt as _;
use crate::traits::query::normalize::AtExt as _;
use on_unimplemented::InferCtxtExt as _;
use suggestions::InferCtxtExt as _;

pub use rustc_infer::traits::error_reporting::*;

pub trait InferCtxtExt<'tcx> {
    fn report_fulfillment_errors(
        &self,
        errors: &[FulfillmentError<'tcx>],
        body_id: Option<hir::BodyId>,
        fallback_has_occurred: bool,
    );

    fn report_overflow_error<T>(
        &self,
        obligation: &Obligation<'tcx, T>,
        suggest_increasing_limit: bool,
    ) -> !
    where
        T: fmt::Display + TypeFoldable<'tcx>;

    fn report_overflow_error_cycle(&self, cycle: &[PredicateObligation<'tcx>]) -> !;

    /// The `root_obligation` parameter should be the `root_obligation` field
    /// from a `FulfillmentError`. If no `FulfillmentError` is available,
    /// then it should be the same as `obligation`.
    fn report_selection_error(
        &self,
        obligation: PredicateObligation<'tcx>,
        root_obligation: &PredicateObligation<'tcx>,
        error: &SelectionError<'tcx>,
        fallback_has_occurred: bool,
    );

    /// Given some node representing a fn-like thing in the HIR map,
    /// returns a span and `ArgKind` information that describes the
    /// arguments it expects. This can be supplied to
    /// `report_arg_count_mismatch`.
    fn get_fn_like_arguments(&self, node: Node<'_>) -> Option<(Span, Vec<ArgKind>)>;

    /// Reports an error when the number of arguments needed by a
    /// trait match doesn't match the number that the expression
    /// provides.
    fn report_arg_count_mismatch(
        &self,
        span: Span,
        found_span: Option<Span>,
        expected_args: Vec<ArgKind>,
        found_args: Vec<ArgKind>,
        is_closure: bool,
    ) -> DiagnosticBuilder<'tcx>;
}

impl<'a, 'tcx> InferCtxtExt<'tcx> for InferCtxt<'a, 'tcx> {
    fn report_fulfillment_errors(
        &self,
        errors: &[FulfillmentError<'tcx>],
        body_id: Option<hir::BodyId>,
        fallback_has_occurred: bool,
    ) {
        #[derive(Debug)]
        struct ErrorDescriptor<'tcx> {
            predicate: ty::Predicate<'tcx>,
            index: Option<usize>, // None if this is an old error
        }

        let mut error_map: FxHashMap<_, Vec<_>> = self
            .reported_trait_errors
            .borrow()
            .iter()
            .map(|(&span, predicates)| {
                (
                    span,
                    predicates
                        .iter()
                        .map(|&predicate| ErrorDescriptor { predicate, index: None })
                        .collect(),
                )
            })
            .collect();

        for (index, error) in errors.iter().enumerate() {
            // We want to ignore desugarings here: spans are equivalent even
            // if one is the result of a desugaring and the other is not.
            let mut span = error.obligation.cause.span;
            let expn_data = span.ctxt().outer_expn_data();
            if let ExpnKind::Desugaring(_) = expn_data.kind {
                span = expn_data.call_site;
            }

            error_map.entry(span).or_default().push(ErrorDescriptor {
                predicate: error.obligation.predicate,
                index: Some(index),
            });

            self.reported_trait_errors
                .borrow_mut()
                .entry(span)
                .or_default()
                .push(error.obligation.predicate);
        }

        // We do this in 2 passes because we want to display errors in order, though
        // maybe it *is* better to sort errors by span or something.
        let mut is_suppressed = vec![false; errors.len()];
        for (_, error_set) in error_map.iter() {
            // We want to suppress "duplicate" errors with the same span.
            for error in error_set {
                if let Some(index) = error.index {
                    // Suppress errors that are either:
                    // 1) strictly implied by another error.
                    // 2) implied by an error with a smaller index.
                    for error2 in error_set {
                        if error2.index.map_or(false, |index2| is_suppressed[index2]) {
                            // Avoid errors being suppressed by already-suppressed
                            // errors, to prevent all errors from being suppressed
                            // at once.
                            continue;
                        }

                        if self.error_implies(error2.predicate, error.predicate)
                            && !(error2.index >= error.index
                                && self.error_implies(error.predicate, error2.predicate))
                        {
                            info!("skipping {:?} (implied by {:?})", error, error2);
                            is_suppressed[index] = true;
                            break;
                        }
                    }
                }
            }
        }

        for (error, suppressed) in iter::zip(errors, is_suppressed) {
            if !suppressed {
                self.report_fulfillment_error(error, body_id, fallback_has_occurred);
            }
        }
    }

    /// Reports that an overflow has occurred and halts compilation. We
    /// halt compilation unconditionally because it is important that
    /// overflows never be masked -- they basically represent computations
    /// whose result could not be truly determined and thus we can't say
    /// if the program type checks or not -- and they are unusual
    /// occurrences in any case.
    fn report_overflow_error<T>(
        &self,
        obligation: &Obligation<'tcx, T>,
        suggest_increasing_limit: bool,
    ) -> !
    where
        T: fmt::Display + TypeFoldable<'tcx>,
    {
        let predicate = self.resolve_vars_if_possible(obligation.predicate.clone());
        let mut err = struct_span_err!(
            self.tcx.sess,
            obligation.cause.span,
            E0275,
            "overflow evaluating the requirement `{}`",
            predicate
        );

        if suggest_increasing_limit {
            self.suggest_new_overflow_limit(&mut err);
        }

        self.note_obligation_cause_code(
            &mut err,
            &obligation.predicate,
            &obligation.cause.code,
            &mut vec![],
            &mut Default::default(),
        );

        err.emit();
        self.tcx.sess.abort_if_errors();
        bug!();
    }

    /// Reports that a cycle was detected which led to overflow and halts
    /// compilation. This is equivalent to `report_overflow_error` except
    /// that we can give a more helpful error message (and, in particular,
    /// we do not suggest increasing the overflow limit, which is not
    /// going to help).
    fn report_overflow_error_cycle(&self, cycle: &[PredicateObligation<'tcx>]) -> ! {
        let cycle = self.resolve_vars_if_possible(cycle.to_owned());
        assert!(!cycle.is_empty());

        debug!("report_overflow_error_cycle: cycle={:?}", cycle);

        // The 'deepest' obligation is most likely to have a useful
        // cause 'backtrace'
        self.report_overflow_error(cycle.iter().max_by_key(|p| p.recursion_depth).unwrap(), false);
    }

    fn report_selection_error(
        &self,
        mut obligation: PredicateObligation<'tcx>,
        root_obligation: &PredicateObligation<'tcx>,
        error: &SelectionError<'tcx>,
        fallback_has_occurred: bool,
    ) {
        let tcx = self.tcx;
        let mut span = obligation.cause.span;

        let mut err = match *error {
            SelectionError::Unimplemented => {
                // If this obligation was generated as a result of well-formedness checking, see if we
                // can get a better error message by performing HIR-based well-formedness checking.
                if let ObligationCauseCode::WellFormed(Some(wf_loc)) =
                    root_obligation.cause.code.peel_derives()
                {
                    if let Some(cause) = self
                        .tcx
                        .diagnostic_hir_wf_check((tcx.erase_regions(obligation.predicate), *wf_loc))
                    {
                        obligation.cause = cause;
                        span = obligation.cause.span;
                    }
                }
                if let ObligationCauseCode::CompareImplMethodObligation {
                    item_name,
                    impl_item_def_id,
                    trait_item_def_id,
                }
                | ObligationCauseCode::CompareImplTypeObligation {
                    item_name,
                    impl_item_def_id,
                    trait_item_def_id,
                } = obligation.cause.code
                {
                    self.report_extra_impl_obligation(
                        span,
                        item_name,
                        impl_item_def_id,
                        trait_item_def_id,
                        &format!("`{}`", obligation.predicate),
                    )
                    .emit();
                    return;
                }

                let bound_predicate = obligation.predicate.kind();
                match bound_predicate.skip_binder() {
                    ty::PredicateKind::Trait(trait_predicate) => {
                        let trait_predicate = bound_predicate.rebind(trait_predicate);
                        let trait_predicate = self.resolve_vars_if_possible(trait_predicate);

                        if self.tcx.sess.has_errors() && trait_predicate.references_error() {
                            return;
                        }
                        let trait_ref = trait_predicate.to_poly_trait_ref();
                        let (post_message, pre_message, type_def) = self
                            .get_parent_trait_ref(&obligation.cause.code)
                            .map(|(t, s)| {
                                (
                                    format!(" in `{}`", t),
                                    format!("within `{}`, ", t),
                                    s.map(|s| (format!("within this `{}`", t), s)),
                                )
                            })
                            .unwrap_or_default();

                        let OnUnimplementedNote { message, label, note, enclosing_scope } =
                            self.on_unimplemented_note(trait_ref, &obligation);
                        let have_alt_message = message.is_some() || label.is_some();
                        let is_try_conversion = self.is_try_conversion(span, trait_ref.def_id());
                        let is_unsize =
                            { Some(trait_ref.def_id()) == self.tcx.lang_items().unsize_trait() };
                        let (message, note) = if is_try_conversion {
                            (
                                Some(format!(
                                    "`?` couldn't convert the error to `{}`",
                                    trait_ref.skip_binder().self_ty(),
                                )),
                                Some(
                                    "the question mark operation (`?`) implicitly performs a \
                                        conversion on the error value using the `From` trait"
                                        .to_owned(),
                                ),
                            )
                        } else {
                            (message, note)
                        };

                        let mut err = struct_span_err!(
                            self.tcx.sess,
                            span,
                            E0277,
                            "{}",
                            message.unwrap_or_else(|| format!(
                                "the trait bound `{}` is not satisfied{}",
                                trait_ref.without_const().to_predicate(tcx),
                                post_message,
                            ))
                        );

                        if is_try_conversion {
                            let none_error = self
                                .tcx
                                .get_diagnostic_item(sym::none_error)
                                .map(|def_id| tcx.type_of(def_id));
                            let should_convert_option_to_result =
                                Some(trait_ref.skip_binder().substs.type_at(1)) == none_error;
                            let should_convert_result_to_option =
                                Some(trait_ref.self_ty().skip_binder()) == none_error;
                            if should_convert_option_to_result {
                                err.span_suggestion_verbose(
                                    span.shrink_to_lo(),
                                    "consider converting the `Option<T>` into a `Result<T, _>` \
                                     using `Option::ok_or` or `Option::ok_or_else`",
                                    ".ok_or_else(|| /* error value */)".to_string(),
                                    Applicability::HasPlaceholders,
                                );
                            } else if should_convert_result_to_option {
                                err.span_suggestion_verbose(
                                    span.shrink_to_lo(),
                                    "consider converting the `Result<T, _>` into an `Option<T>` \
                                     using `Result::ok`",
                                    ".ok()".to_string(),
                                    Applicability::MachineApplicable,
                                );
                            }
                            if let Some(ret_span) = self.return_type_span(&obligation) {
                                err.span_label(
                                    ret_span,
                                    &format!(
                                        "expected `{}` because of this",
                                        trait_ref.skip_binder().self_ty()
                                    ),
                                );
                            }
                        }

                        let explanation =
                            if obligation.cause.code == ObligationCauseCode::MainFunctionType {
                                "consider using `()`, or a `Result`".to_owned()
                            } else {
                                format!(
                                    "{}the trait `{}` is not implemented for `{}`",
                                    pre_message,
                                    trait_ref.print_only_trait_path(),
                                    trait_ref.skip_binder().self_ty(),
                                )
                            };

                        if self.suggest_add_reference_to_arg(
                            &obligation,
                            &mut err,
                            &trait_ref,
                            have_alt_message,
                        ) {
                            self.note_obligation_cause(&mut err, &obligation);
                            err.emit();
                            return;
                        }
                        if let Some(ref s) = label {
                            // If it has a custom `#[rustc_on_unimplemented]`
                            // error message, let's display it as the label!
                            err.span_label(span, s.as_str());
                            if !matches!(trait_ref.skip_binder().self_ty().kind(), ty::Param(_)) {
                                // When the self type is a type param We don't need to "the trait
                                // `std::marker::Sized` is not implemented for `T`" as we will point
                                // at the type param with a label to suggest constraining it.
                                err.help(&explanation);
                            }
                        } else {
                            err.span_label(span, explanation);
                        }
                        if let Some((msg, span)) = type_def {
                            err.span_label(span, &msg);
                        }
                        if let Some(ref s) = note {
                            // If it has a custom `#[rustc_on_unimplemented]` note, let's display it
                            err.note(s.as_str());
                        }
                        if let Some(ref s) = enclosing_scope {
                            let body = tcx
                                .hir()
                                .opt_local_def_id(obligation.cause.body_id)
                                .unwrap_or_else(|| {
                                    tcx.hir().body_owner_def_id(hir::BodyId {
                                        hir_id: obligation.cause.body_id,
                                    })
                                });

                            let enclosing_scope_span =
                                tcx.hir().span_with_body(tcx.hir().local_def_id_to_hir_id(body));

                            err.span_label(enclosing_scope_span, s.as_str());
                        }

                        self.suggest_dereferences(&obligation, &mut err, trait_ref);
                        self.suggest_fn_call(&obligation, &mut err, trait_ref);
                        self.suggest_remove_reference(&obligation, &mut err, trait_ref);
                        self.suggest_semicolon_removal(&obligation, &mut err, span, trait_ref);
                        self.note_version_mismatch(&mut err, &trait_ref);

                        if Some(trait_ref.def_id()) == tcx.lang_items().try_trait() {
                            self.suggest_await_before_try(&mut err, &obligation, trait_ref, span);
                        }

                        if self.suggest_impl_trait(&mut err, span, &obligation, trait_ref) {
                            err.emit();
                            return;
                        }

                        if is_unsize {
                            // If the obligation failed due to a missing implementation of the
                            // `Unsize` trait, give a pointer to why that might be the case
                            err.note(
                                "all implementations of `Unsize` are provided \
                                automatically by the compiler, see \
                                <https://doc.rust-lang.org/stable/std/marker/trait.Unsize.html> \
                                for more information",
                            );
                        }

                        let is_fn_trait = [
                            self.tcx.lang_items().fn_trait(),
                            self.tcx.lang_items().fn_mut_trait(),
                            self.tcx.lang_items().fn_once_trait(),
                        ]
                        .contains(&Some(trait_ref.def_id()));
                        let is_target_feature_fn = if let ty::FnDef(def_id, _) =
                            *trait_ref.skip_binder().self_ty().kind()
                        {
                            !self.tcx.codegen_fn_attrs(def_id).target_features.is_empty()
                        } else {
                            false
                        };
                        if is_fn_trait && is_target_feature_fn {
                            err.note(
                                "`#[target_feature]` functions do not implement the `Fn` traits",
                            );
                        }

                        // Try to report a help message
                        if !trait_ref.has_infer_types_or_consts()
                            && self.predicate_can_apply(obligation.param_env, trait_ref)
                        {
                            // If a where-clause may be useful, remind the
                            // user that they can add it.
                            //
                            // don't display an on-unimplemented note, as
                            // these notes will often be of the form
                            //     "the type `T` can't be frobnicated"
                            // which is somewhat confusing.
                            self.suggest_restricting_param_bound(
                                &mut err,
                                trait_ref,
                                obligation.cause.body_id,
                            );
                        } else if !have_alt_message {
                            // Can't show anything else useful, try to find similar impls.
                            let impl_candidates = self.find_similar_impl_candidates(trait_ref);
                            self.report_similar_impl_candidates(impl_candidates, &mut err);
                        }

                        // Changing mutability doesn't make a difference to whether we have
                        // an `Unsize` impl (Fixes ICE in #71036)
                        if !is_unsize {
                            self.suggest_change_mut(&obligation, &mut err, trait_ref);
                        }

                        // If this error is due to `!: Trait` not implemented but `(): Trait` is
                        // implemented, and fallback has occurred, then it could be due to a
                        // variable that used to fallback to `()` now falling back to `!`. Issue a
                        // note informing about the change in behaviour.
                        if trait_predicate.skip_binder().self_ty().is_never()
                            && fallback_has_occurred
                        {
                            let predicate = trait_predicate.map_bound(|mut trait_pred| {
                                trait_pred.trait_ref.substs = self.tcx.mk_substs_trait(
                                    self.tcx.mk_unit(),
                                    &trait_pred.trait_ref.substs[1..],
                                );
                                trait_pred
                            });
                            let unit_obligation = obligation.with(predicate.to_predicate(tcx));
                            if self.predicate_may_hold(&unit_obligation) {
                                err.note("this trait is implemented for `()`");
                                err.note(
                                    "this error might have been caused by changes to \
                                    Rust's type-inference algorithm (see issue #48950 \
                                    <https://github.com/rust-lang/rust/issues/48950> \
                                    for more information).",
                                );
                                err.help("did you intend to use the type `()` here instead?");
                            }
                        }

                        // Return early if the trait is Debug or Display and the invocation
                        // originates within a standard library macro, because the output
                        // is otherwise overwhelming and unhelpful (see #85844 for an
                        // example).

                        let trait_is_debug =
                            self.tcx.is_diagnostic_item(sym::Debug, trait_ref.def_id());
                        let trait_is_display =
                            self.tcx.is_diagnostic_item(sym::Display, trait_ref.def_id());

                        let in_std_macro =
                            match obligation.cause.span.ctxt().outer_expn_data().macro_def_id {
                                Some(macro_def_id) => {
                                    let crate_name = tcx.crate_name(macro_def_id.krate);
                                    crate_name == sym::std || crate_name == sym::core
                                }
                                None => false,
                            };

                        if in_std_macro && (trait_is_debug || trait_is_display) {
                            err.emit();
                            return;
                        }

                        err
                    }

                    ty::PredicateKind::Subtype(predicate) => {
                        // Errors for Subtype predicates show up as
                        // `FulfillmentErrorCode::CodeSubtypeError`,
                        // not selection error.
                        span_bug!(span, "subtype requirement gave wrong error: `{:?}`", predicate)
                    }

                    ty::PredicateKind::Coerce(predicate) => {
                        // Errors for Coerce predicates show up as
                        // `FulfillmentErrorCode::CodeSubtypeError`,
                        // not selection error.
                        span_bug!(span, "coerce requirement gave wrong error: `{:?}`", predicate)
                    }

                    ty::PredicateKind::RegionOutlives(predicate) => {
                        let predicate = bound_predicate.rebind(predicate);
                        let predicate = self.resolve_vars_if_possible(predicate);
                        let err = self
                            .region_outlives_predicate(&obligation.cause, predicate)
                            .err()
                            .unwrap();
                        struct_span_err!(
                            self.tcx.sess,
                            span,
                            E0279,
                            "the requirement `{}` is not satisfied (`{}`)",
                            predicate,
                            err,
                        )
                    }

                    ty::PredicateKind::Projection(..) | ty::PredicateKind::TypeOutlives(..) => {
                        let predicate = self.resolve_vars_if_possible(obligation.predicate);
                        struct_span_err!(
                            self.tcx.sess,
                            span,
                            E0280,
                            "the requirement `{}` is not satisfied",
                            predicate
                        )
                    }

                    ty::PredicateKind::ObjectSafe(trait_def_id) => {
                        let violations = self.tcx.object_safety_violations(trait_def_id);
                        report_object_safety_error(self.tcx, span, trait_def_id, violations)
                    }

                    ty::PredicateKind::ClosureKind(closure_def_id, closure_substs, kind) => {
                        let found_kind = self.closure_kind(closure_substs).unwrap();
                        let closure_span =
                            self.tcx.sess.source_map().guess_head_span(
                                self.tcx.hir().span_if_local(closure_def_id).unwrap(),
                            );
                        let hir_id =
                            self.tcx.hir().local_def_id_to_hir_id(closure_def_id.expect_local());
                        let mut err = struct_span_err!(
                            self.tcx.sess,
                            closure_span,
                            E0525,
                            "expected a closure that implements the `{}` trait, \
                             but this closure only implements `{}`",
                            kind,
                            found_kind
                        );

                        err.span_label(
                            closure_span,
                            format!("this closure implements `{}`, not `{}`", found_kind, kind),
                        );
                        err.span_label(
                            obligation.cause.span,
                            format!("the requirement to implement `{}` derives from here", kind),
                        );

                        // Additional context information explaining why the closure only implements
                        // a particular trait.
                        if let Some(typeck_results) = self.in_progress_typeck_results {
                            let typeck_results = typeck_results.borrow();
                            match (found_kind, typeck_results.closure_kind_origins().get(hir_id)) {
                                (ty::ClosureKind::FnOnce, Some((span, place))) => {
                                    err.span_label(
                                        *span,
                                        format!(
                                            "closure is `FnOnce` because it moves the \
                                         variable `{}` out of its environment",
                                            ty::place_to_string_for_capture(tcx, place)
                                        ),
                                    );
                                }
                                (ty::ClosureKind::FnMut, Some((span, place))) => {
                                    err.span_label(
                                        *span,
                                        format!(
                                            "closure is `FnMut` because it mutates the \
                                         variable `{}` here",
                                            ty::place_to_string_for_capture(tcx, place)
                                        ),
                                    );
                                }
                                _ => {}
                            }
                        }

                        err.emit();
                        return;
                    }

                    ty::PredicateKind::WellFormed(ty) => {
                        if !self.tcx.sess.opts.debugging_opts.chalk {
                            // WF predicates cannot themselves make
                            // errors. They can only block due to
                            // ambiguity; otherwise, they always
                            // degenerate into other obligations
                            // (which may fail).
                            span_bug!(span, "WF predicate not satisfied for {:?}", ty);
                        } else {
                            // FIXME: we'll need a better message which takes into account
                            // which bounds actually failed to hold.
                            self.tcx.sess.struct_span_err(
                                span,
                                &format!("the type `{}` is not well-formed (chalk)", ty),
                            )
                        }
                    }

                    ty::PredicateKind::ConstEvaluatable(..) => {
                        // Errors for `ConstEvaluatable` predicates show up as
                        // `SelectionError::ConstEvalFailure`,
                        // not `Unimplemented`.
                        span_bug!(
                            span,
                            "const-evaluatable requirement gave wrong error: `{:?}`",
                            obligation
                        )
                    }

                    ty::PredicateKind::ConstEquate(..) => {
                        // Errors for `ConstEquate` predicates show up as
                        // `SelectionError::ConstEvalFailure`,
                        // not `Unimplemented`.
                        span_bug!(
                            span,
                            "const-equate requirement gave wrong error: `{:?}`",
                            obligation
                        )
                    }

                    ty::PredicateKind::TypeWellFormedFromEnv(..) => span_bug!(
                        span,
                        "TypeWellFormedFromEnv predicate should only exist in the environment"
                    ),
                }
            }

            OutputTypeParameterMismatch(found_trait_ref, expected_trait_ref, _) => {
                let found_trait_ref = self.resolve_vars_if_possible(found_trait_ref);
                let expected_trait_ref = self.resolve_vars_if_possible(expected_trait_ref);

                if expected_trait_ref.self_ty().references_error() {
                    return;
                }

                let found_trait_ty = match found_trait_ref.self_ty().no_bound_vars() {
                    Some(ty) => ty,
                    None => return,
                };

                let found_did = match *found_trait_ty.kind() {
                    ty::Closure(did, _)
                    | ty::Foreign(did)
                    | ty::FnDef(did, _)
                    | ty::Generator(did, ..) => Some(did),
                    ty::Adt(def, _) => Some(def.did),
                    _ => None,
                };

                let found_span = found_did
                    .and_then(|did| self.tcx.hir().span_if_local(did))
                    .map(|sp| self.tcx.sess.source_map().guess_head_span(sp)); // the sp could be an fn def

                if self.reported_closure_mismatch.borrow().contains(&(span, found_span)) {
                    // We check closures twice, with obligations flowing in different directions,
                    // but we want to complain about them only once.
                    return;
                }

                self.reported_closure_mismatch.borrow_mut().insert((span, found_span));

                let found = match found_trait_ref.skip_binder().substs.type_at(1).kind() {
                    ty::Tuple(ref tys) => vec![ArgKind::empty(); tys.len()],
                    _ => vec![ArgKind::empty()],
                };

                let expected_ty = expected_trait_ref.skip_binder().substs.type_at(1);
                let expected = match expected_ty.kind() {
                    ty::Tuple(ref tys) => tys
                        .iter()
                        .map(|t| ArgKind::from_expected_ty(t.expect_ty(), Some(span)))
                        .collect(),
                    _ => vec![ArgKind::Arg("_".to_owned(), expected_ty.to_string())],
                };

                if found.len() == expected.len() {
                    self.report_closure_arg_mismatch(
                        span,
                        found_span,
                        found_trait_ref,
                        expected_trait_ref,
                    )
                } else {
                    let (closure_span, found) = found_did
                        .and_then(|did| {
                            let node = self.tcx.hir().get_if_local(did)?;
                            let (found_span, found) = self.get_fn_like_arguments(node)?;
                            Some((Some(found_span), found))
                        })
                        .unwrap_or((found_span, found));

                    self.report_arg_count_mismatch(
                        span,
                        closure_span,
                        expected,
                        found,
                        found_trait_ty.is_closure(),
                    )
                }
            }

            TraitNotObjectSafe(did) => {
                let violations = self.tcx.object_safety_violations(did);
                report_object_safety_error(self.tcx, span, did, violations)
            }

            SelectionError::NotConstEvaluatable(NotConstEvaluatable::MentionsInfer) => {
                bug!(
                    "MentionsInfer should have been handled in `traits/fulfill.rs` or `traits/select/mod.rs`"
                )
            }
            SelectionError::NotConstEvaluatable(NotConstEvaluatable::MentionsParam) => {
                if !self.tcx.features().generic_const_exprs {
                    let mut err = self.tcx.sess.struct_span_err(
                        span,
                        "constant expression depends on a generic parameter",
                    );
                    // FIXME(const_generics): we should suggest to the user how they can resolve this
                    // issue. However, this is currently not actually possible
                    // (see https://github.com/rust-lang/rust/issues/66962#issuecomment-575907083).
                    //
                    // Note that with `feature(generic_const_exprs)` this case should not
                    // be reachable.
                    err.note("this may fail depending on what value the parameter takes");
                    err.emit();
                    return;
                }

                match obligation.predicate.kind().skip_binder() {
                    ty::PredicateKind::ConstEvaluatable(uv) => {
                        let mut err =
                            self.tcx.sess.struct_span_err(span, "unconstrained generic constant");
                        let const_span = self.tcx.def_span(uv.def.did);
                        match self.tcx.sess.source_map().span_to_snippet(const_span) {
                            Ok(snippet) => err.help(&format!(
                                "try adding a `where` bound using this expression: `where [(); {}]:`",
                                snippet
                            )),
                            _ => err.help("consider adding a `where` bound using this expression"),
                        };
                        err
                    }
                    _ => {
                        span_bug!(
                            span,
                            "unexpected non-ConstEvaluatable predicate, this should not be reachable"
                        )
                    }
                }
            }

            // Already reported in the query.
            SelectionError::NotConstEvaluatable(NotConstEvaluatable::Error(ErrorReported)) => {
                // FIXME(eddyb) remove this once `ErrorReported` becomes a proof token.
                self.tcx.sess.delay_span_bug(span, "`ErrorReported` without an error");
                return;
            }

            Overflow => {
                bug!("overflow should be handled before the `report_selection_error` path");
            }
        };

        self.note_obligation_cause(&mut err, &obligation);
        self.point_at_returns_when_relevant(&mut err, &obligation);

        err.emit();
    }

    /// Given some node representing a fn-like thing in the HIR map,
    /// returns a span and `ArgKind` information that describes the
    /// arguments it expects. This can be supplied to
    /// `report_arg_count_mismatch`.
    fn get_fn_like_arguments(&self, node: Node<'_>) -> Option<(Span, Vec<ArgKind>)> {
        let sm = self.tcx.sess.source_map();
        let hir = self.tcx.hir();
        Some(match node {
            Node::Expr(&hir::Expr {
                kind: hir::ExprKind::Closure(_, ref _decl, id, span, _),
                ..
            }) => (
                sm.guess_head_span(span),
                hir.body(id)
                    .params
                    .iter()
                    .map(|arg| {
                        if let hir::Pat { kind: hir::PatKind::Tuple(ref args, _), span, .. } =
                            *arg.pat
                        {
                            Some(ArgKind::Tuple(
                                Some(span),
                                args.iter()
                                    .map(|pat| {
                                        sm.span_to_snippet(pat.span)
                                            .ok()
                                            .map(|snippet| (snippet, "_".to_owned()))
                                    })
                                    .collect::<Option<Vec<_>>>()?,
                            ))
                        } else {
                            let name = sm.span_to_snippet(arg.pat.span).ok()?;
                            Some(ArgKind::Arg(name, "_".to_owned()))
                        }
                    })
                    .collect::<Option<Vec<ArgKind>>>()?,
            ),
            Node::Item(&hir::Item { span, kind: hir::ItemKind::Fn(ref sig, ..), .. })
            | Node::ImplItem(&hir::ImplItem {
                span,
                kind: hir::ImplItemKind::Fn(ref sig, _),
                ..
            })
            | Node::TraitItem(&hir::TraitItem {
                span,
                kind: hir::TraitItemKind::Fn(ref sig, _),
                ..
            }) => (
                sm.guess_head_span(span),
                sig.decl
                    .inputs
                    .iter()
                    .map(|arg| match arg.kind {
                        hir::TyKind::Tup(ref tys) => ArgKind::Tuple(
                            Some(arg.span),
                            vec![("_".to_owned(), "_".to_owned()); tys.len()],
                        ),
                        _ => ArgKind::empty(),
                    })
                    .collect::<Vec<ArgKind>>(),
            ),
            Node::Ctor(ref variant_data) => {
                let span = variant_data.ctor_hir_id().map_or(DUMMY_SP, |id| hir.span(id));
                let span = sm.guess_head_span(span);
                (span, vec![ArgKind::empty(); variant_data.fields().len()])
            }
            _ => panic!("non-FnLike node found: {:?}", node),
        })
    }

    /// Reports an error when the number of arguments needed by a
    /// trait match doesn't match the number that the expression
    /// provides.
    fn report_arg_count_mismatch(
        &self,
        span: Span,
        found_span: Option<Span>,
        expected_args: Vec<ArgKind>,
        found_args: Vec<ArgKind>,
        is_closure: bool,
    ) -> DiagnosticBuilder<'tcx> {
        let kind = if is_closure { "closure" } else { "function" };

        let args_str = |arguments: &[ArgKind], other: &[ArgKind]| {
            let arg_length = arguments.len();
            let distinct = matches!(other, &[ArgKind::Tuple(..)]);
            match (arg_length, arguments.get(0)) {
                (1, Some(&ArgKind::Tuple(_, ref fields))) => {
                    format!("a single {}-tuple as argument", fields.len())
                }
                _ => format!(
                    "{} {}argument{}",
                    arg_length,
                    if distinct && arg_length > 1 { "distinct " } else { "" },
                    pluralize!(arg_length)
                ),
            }
        };

        let expected_str = args_str(&expected_args, &found_args);
        let found_str = args_str(&found_args, &expected_args);

        let mut err = struct_span_err!(
            self.tcx.sess,
            span,
            E0593,
            "{} is expected to take {}, but it takes {}",
            kind,
            expected_str,
            found_str,
        );

        err.span_label(span, format!("expected {} that takes {}", kind, expected_str));

        if let Some(found_span) = found_span {
            err.span_label(found_span, format!("takes {}", found_str));

            // move |_| { ... }
            // ^^^^^^^^-- def_span
            //
            // move |_| { ... }
            // ^^^^^-- prefix
            let prefix_span = self.tcx.sess.source_map().span_until_non_whitespace(found_span);
            // move |_| { ... }
            //      ^^^-- pipe_span
            let pipe_span =
                if let Some(span) = found_span.trim_start(prefix_span) { span } else { found_span };

            // Suggest to take and ignore the arguments with expected_args_length `_`s if
            // found arguments is empty (assume the user just wants to ignore args in this case).
            // For example, if `expected_args_length` is 2, suggest `|_, _|`.
            if found_args.is_empty() && is_closure {
                let underscores = vec!["_"; expected_args.len()].join(", ");
                err.span_suggestion_verbose(
                    pipe_span,
                    &format!(
                        "consider changing the closure to take and ignore the expected argument{}",
                        pluralize!(expected_args.len())
                    ),
                    format!("|{}|", underscores),
                    Applicability::MachineApplicable,
                );
            }

            if let &[ArgKind::Tuple(_, ref fields)] = &found_args[..] {
                if fields.len() == expected_args.len() {
                    let sugg = fields
                        .iter()
                        .map(|(name, _)| name.to_owned())
                        .collect::<Vec<String>>()
                        .join(", ");
                    err.span_suggestion_verbose(
                        found_span,
                        "change the closure to take multiple arguments instead of a single tuple",
                        format!("|{}|", sugg),
                        Applicability::MachineApplicable,
                    );
                }
            }
            if let &[ArgKind::Tuple(_, ref fields)] = &expected_args[..] {
                if fields.len() == found_args.len() && is_closure {
                    let sugg = format!(
                        "|({}){}|",
                        found_args
                            .iter()
                            .map(|arg| match arg {
                                ArgKind::Arg(name, _) => name.to_owned(),
                                _ => "_".to_owned(),
                            })
                            .collect::<Vec<String>>()
                            .join(", "),
                        // add type annotations if available
                        if found_args.iter().any(|arg| match arg {
                            ArgKind::Arg(_, ty) => ty != "_",
                            _ => false,
                        }) {
                            format!(
                                ": ({})",
                                fields
                                    .iter()
                                    .map(|(_, ty)| ty.to_owned())
                                    .collect::<Vec<String>>()
                                    .join(", ")
                            )
                        } else {
                            String::new()
                        },
                    );
                    err.span_suggestion_verbose(
                        found_span,
                        "change the closure to accept a tuple instead of individual arguments",
                        sugg,
                        Applicability::MachineApplicable,
                    );
                }
            }
        }

        err
    }
}

trait InferCtxtPrivExt<'tcx> {
    // returns if `cond` not occurring implies that `error` does not occur - i.e., that
    // `error` occurring implies that `cond` occurs.
    fn error_implies(&self, cond: ty::Predicate<'tcx>, error: ty::Predicate<'tcx>) -> bool;

    fn report_fulfillment_error(
        &self,
        error: &FulfillmentError<'tcx>,
        body_id: Option<hir::BodyId>,
        fallback_has_occurred: bool,
    );

    fn report_projection_error(
        &self,
        obligation: &PredicateObligation<'tcx>,
        error: &MismatchedProjectionTypes<'tcx>,
    );

    fn fuzzy_match_tys(&self, a: Ty<'tcx>, b: Ty<'tcx>) -> bool;

    fn describe_generator(&self, body_id: hir::BodyId) -> Option<&'static str>;

    fn find_similar_impl_candidates(
        &self,
        trait_ref: ty::PolyTraitRef<'tcx>,
    ) -> Vec<ty::TraitRef<'tcx>>;

    fn report_similar_impl_candidates(
        &self,
        impl_candidates: Vec<ty::TraitRef<'tcx>>,
        err: &mut DiagnosticBuilder<'_>,
    );

    /// Gets the parent trait chain start
    fn get_parent_trait_ref(
        &self,
        code: &ObligationCauseCode<'tcx>,
    ) -> Option<(String, Option<Span>)>;

    /// If the `Self` type of the unsatisfied trait `trait_ref` implements a trait
    /// with the same path as `trait_ref`, a help message about
    /// a probable version mismatch is added to `err`
    fn note_version_mismatch(
        &self,
        err: &mut DiagnosticBuilder<'_>,
        trait_ref: &ty::PolyTraitRef<'tcx>,
    );

    /// Creates a `PredicateObligation` with `new_self_ty` replacing the existing type in the
    /// `trait_ref`.
    ///
    /// For this to work, `new_self_ty` must have no escaping bound variables.
    fn mk_trait_obligation_with_new_self_ty(
        &self,
        param_env: ty::ParamEnv<'tcx>,
        trait_ref: ty::PolyTraitRef<'tcx>,
        new_self_ty: Ty<'tcx>,
    ) -> PredicateObligation<'tcx>;

    fn maybe_report_ambiguity(
        &self,
        obligation: &PredicateObligation<'tcx>,
        body_id: Option<hir::BodyId>,
    );

    fn predicate_can_apply(
        &self,
        param_env: ty::ParamEnv<'tcx>,
        pred: ty::PolyTraitRef<'tcx>,
    ) -> bool;

    fn note_obligation_cause(
        &self,
        err: &mut DiagnosticBuilder<'tcx>,
        obligation: &PredicateObligation<'tcx>,
    );

    fn suggest_unsized_bound_if_applicable(
        &self,
        err: &mut DiagnosticBuilder<'tcx>,
        obligation: &PredicateObligation<'tcx>,
    );

    fn maybe_suggest_unsized_generics(
        &self,
        err: &mut DiagnosticBuilder<'tcx>,
        span: Span,
        node: Node<'hir>,
    );

    fn maybe_indirection_for_unsized(
        &self,
        err: &mut DiagnosticBuilder<'tcx>,
        item: &'hir Item<'hir>,
        param: &'hir GenericParam<'hir>,
    ) -> bool;

    fn is_recursive_obligation(
        &self,
        obligated_types: &mut Vec<&ty::TyS<'tcx>>,
        cause_code: &ObligationCauseCode<'tcx>,
    ) -> bool;
}

impl<'a, 'tcx> InferCtxtPrivExt<'tcx> for InferCtxt<'a, 'tcx> {
    // returns if `cond` not occurring implies that `error` does not occur - i.e., that
    // `error` occurring implies that `cond` occurs.
    fn error_implies(&self, cond: ty::Predicate<'tcx>, error: ty::Predicate<'tcx>) -> bool {
        if cond == error {
            return true;
        }

        // FIXME: It should be possible to deal with `ForAll` in a cleaner way.
        let bound_error = error.kind();
        let (cond, error) = match (cond.kind().skip_binder(), bound_error.skip_binder()) {
            (ty::PredicateKind::Trait(..), ty::PredicateKind::Trait(error)) => {
                (cond, bound_error.rebind(error))
            }
            _ => {
                // FIXME: make this work in other cases too.
                return false;
            }
        };

        for obligation in super::elaborate_predicates(self.tcx, std::iter::once(cond)) {
            let bound_predicate = obligation.predicate.kind();
            if let ty::PredicateKind::Trait(implication) = bound_predicate.skip_binder() {
                let error = error.to_poly_trait_ref();
                let implication = bound_predicate.rebind(implication.trait_ref);
                // FIXME: I'm just not taking associated types at all here.
                // Eventually I'll need to implement param-env-aware
                // `Γ₁ ⊦ φ₁ => Γ₂ ⊦ φ₂` logic.
                let param_env = ty::ParamEnv::empty();
                if self.can_sub(param_env, error, implication).is_ok() {
                    debug!("error_implies: {:?} -> {:?} -> {:?}", cond, error, implication);
                    return true;
                }
            }
        }

        false
    }

    #[instrument(skip(self), level = "debug")]
    fn report_fulfillment_error(
        &self,
        error: &FulfillmentError<'tcx>,
        body_id: Option<hir::BodyId>,
        fallback_has_occurred: bool,
    ) {
        match error.code {
            FulfillmentErrorCode::CodeSelectionError(ref selection_error) => {
                self.report_selection_error(
                    error.obligation.clone(),
                    &error.root_obligation,
                    selection_error,
                    fallback_has_occurred,
                );
            }
            FulfillmentErrorCode::CodeProjectionError(ref e) => {
                self.report_projection_error(&error.obligation, e);
            }
            FulfillmentErrorCode::CodeAmbiguity => {
                self.maybe_report_ambiguity(&error.obligation, body_id);
            }
            FulfillmentErrorCode::CodeSubtypeError(ref expected_found, ref err) => {
                self.report_mismatched_types(
                    &error.obligation.cause,
                    expected_found.expected,
                    expected_found.found,
                    err.clone(),
                )
                .emit();
            }
            FulfillmentErrorCode::CodeConstEquateError(ref expected_found, ref err) => {
                self.report_mismatched_consts(
                    &error.obligation.cause,
                    expected_found.expected,
                    expected_found.found,
                    err.clone(),
                )
                .emit();
            }
        }
    }

    fn report_projection_error(
        &self,
        obligation: &PredicateObligation<'tcx>,
        error: &MismatchedProjectionTypes<'tcx>,
    ) {
        let predicate = self.resolve_vars_if_possible(obligation.predicate);

        if predicate.references_error() {
            return;
        }

        self.probe(|_| {
            let err_buf;
            let mut err = &error.err;
            let mut values = None;

            // try to find the mismatched types to report the error with.
            //
            // this can fail if the problem was higher-ranked, in which
            // cause I have no idea for a good error message.
            let bound_predicate = predicate.kind();
            if let ty::PredicateKind::Projection(data) = bound_predicate.skip_binder() {
                let mut selcx = SelectionContext::new(self);
                let (data, _) = self.replace_bound_vars_with_fresh_vars(
                    obligation.cause.span,
                    infer::LateBoundRegionConversionTime::HigherRankedType,
                    bound_predicate.rebind(data),
                );
                let mut obligations = vec![];
                let normalized_ty = super::normalize_projection_type(
                    &mut selcx,
                    obligation.param_env,
                    data.projection_ty,
                    obligation.cause.clone(),
                    0,
                    &mut obligations,
                );

                debug!(
                    "report_projection_error obligation.cause={:?} obligation.param_env={:?}",
                    obligation.cause, obligation.param_env
                );

                debug!(
                    "report_projection_error normalized_ty={:?} data.ty={:?}",
                    normalized_ty, data.ty
                );

                let is_normalized_ty_expected = !matches!(
                    obligation.cause.code.peel_derives(),
                    ObligationCauseCode::ItemObligation(_)
                        | ObligationCauseCode::BindingObligation(_, _)
                        | ObligationCauseCode::ObjectCastObligation(_)
                        | ObligationCauseCode::OpaqueType
                );

                if let Err(error) = self.at(&obligation.cause, obligation.param_env).eq_exp(
                    is_normalized_ty_expected,
                    normalized_ty,
                    data.ty,
                ) {
                    values = Some(infer::ValuePairs::Types(ExpectedFound::new(
                        is_normalized_ty_expected,
                        normalized_ty,
                        data.ty,
                    )));

                    err_buf = error;
                    err = &err_buf;
                }
            }

            let msg = format!("type mismatch resolving `{}`", predicate);
            let error_id = (DiagnosticMessageId::ErrorId(271), Some(obligation.cause.span), msg);
            let fresh = self.tcx.sess.one_time_diagnostics.borrow_mut().insert(error_id);
            if fresh {
                let mut diag = struct_span_err!(
                    self.tcx.sess,
                    obligation.cause.span,
                    E0271,
                    "type mismatch resolving `{}`",
                    predicate
                );
                self.note_type_err(&mut diag, &obligation.cause, None, values, err);
                self.note_obligation_cause(&mut diag, obligation);
                diag.emit();
            }
        });
    }

    fn fuzzy_match_tys(&self, a: Ty<'tcx>, b: Ty<'tcx>) -> bool {
        /// returns the fuzzy category of a given type, or None
        /// if the type can be equated to any type.
        fn type_category(t: Ty<'_>) -> Option<u32> {
            match t.kind() {
                ty::Bool => Some(0),
                ty::Char => Some(1),
                ty::Str => Some(2),
                ty::Int(..) | ty::Uint(..) | ty::Infer(ty::IntVar(..)) => Some(3),
                ty::Float(..) | ty::Infer(ty::FloatVar(..)) => Some(4),
                ty::Ref(..) | ty::RawPtr(..) => Some(5),
                ty::Array(..) | ty::Slice(..) => Some(6),
                ty::FnDef(..) | ty::FnPtr(..) => Some(7),
                ty::Dynamic(..) => Some(8),
                ty::Closure(..) => Some(9),
                ty::Tuple(..) => Some(10),
                ty::Projection(..) => Some(11),
                ty::Param(..) => Some(12),
                ty::Opaque(..) => Some(13),
                ty::Never => Some(14),
                ty::Adt(adt, ..) => match adt.adt_kind() {
                    AdtKind::Struct => Some(15),
                    AdtKind::Union => Some(16),
                    AdtKind::Enum => Some(17),
                },
                ty::Generator(..) => Some(18),
                ty::Foreign(..) => Some(19),
                ty::GeneratorWitness(..) => Some(20),
                ty::Placeholder(..) | ty::Bound(..) | ty::Infer(..) | ty::Error(_) => None,
            }
        }

        match (type_category(a), type_category(b)) {
            (Some(cat_a), Some(cat_b)) => match (a.kind(), b.kind()) {
                (&ty::Adt(def_a, _), &ty::Adt(def_b, _)) => def_a == def_b,
                _ => cat_a == cat_b,
            },
            // infer and error can be equated to all types
            _ => true,
        }
    }

    fn describe_generator(&self, body_id: hir::BodyId) -> Option<&'static str> {
        self.tcx.hir().body(body_id).generator_kind.map(|gen_kind| match gen_kind {
            hir::GeneratorKind::Gen => "a generator",
            hir::GeneratorKind::Async(hir::AsyncGeneratorKind::Block) => "an async block",
            hir::GeneratorKind::Async(hir::AsyncGeneratorKind::Fn) => "an async function",
            hir::GeneratorKind::Async(hir::AsyncGeneratorKind::Closure) => "an async closure",
        })
    }

    fn find_similar_impl_candidates(
        &self,
        trait_ref: ty::PolyTraitRef<'tcx>,
    ) -> Vec<ty::TraitRef<'tcx>> {
        let simp = fast_reject::simplify_type(self.tcx, trait_ref.skip_binder().self_ty(), true);
        let all_impls = self.tcx.all_impls(trait_ref.def_id());

        match simp {
            Some(simp) => all_impls
                .filter_map(|def_id| {
                    let imp = self.tcx.impl_trait_ref(def_id).unwrap();
                    let imp_simp = fast_reject::simplify_type(self.tcx, imp.self_ty(), true);
                    if let Some(imp_simp) = imp_simp {
                        if simp != imp_simp {
                            return None;
                        }
                    }
                    if self.tcx.impl_polarity(def_id) == ty::ImplPolarity::Negative {
                        return None;
                    }
                    Some(imp)
                })
                .collect(),
            None => all_impls
                .filter_map(|def_id| {
                    if self.tcx.impl_polarity(def_id) == ty::ImplPolarity::Negative {
                        return None;
                    }
                    self.tcx.impl_trait_ref(def_id)
                })
                .collect(),
        }
    }

    fn report_similar_impl_candidates(
        &self,
        impl_candidates: Vec<ty::TraitRef<'tcx>>,
        err: &mut DiagnosticBuilder<'_>,
    ) {
        if impl_candidates.is_empty() {
            return;
        }

        let len = impl_candidates.len();
        let end = if impl_candidates.len() <= 5 { impl_candidates.len() } else { 4 };

        let normalize = |candidate| {
            self.tcx.infer_ctxt().enter(|ref infcx| {
                let normalized = infcx
                    .at(&ObligationCause::dummy(), ty::ParamEnv::empty())
                    .normalize(candidate)
                    .ok();
                match normalized {
                    Some(normalized) => format!("\n  {}", normalized.value),
                    None => format!("\n  {}", candidate),
                }
            })
        };

        // Sort impl candidates so that ordering is consistent for UI tests.
        let mut normalized_impl_candidates =
            impl_candidates.iter().copied().map(normalize).collect::<Vec<String>>();

        // Sort before taking the `..end` range,
        // because the ordering of `impl_candidates` may not be deterministic:
        // https://github.com/rust-lang/rust/pull/57475#issuecomment-455519507
        normalized_impl_candidates.sort();

        err.help(&format!(
            "the following implementations were found:{}{}",
            normalized_impl_candidates[..end].join(""),
            if len > 5 { format!("\nand {} others", len - 4) } else { String::new() }
        ));
    }

    /// Gets the parent trait chain start
    fn get_parent_trait_ref(
        &self,
        code: &ObligationCauseCode<'tcx>,
    ) -> Option<(String, Option<Span>)> {
        match code {
            ObligationCauseCode::BuiltinDerivedObligation(data) => {
                let parent_trait_ref = self.resolve_vars_if_possible(data.parent_trait_ref);
                match self.get_parent_trait_ref(&data.parent_code) {
                    Some(t) => Some(t),
                    None => {
                        let ty = parent_trait_ref.skip_binder().self_ty();
                        let span = TyCategory::from_ty(self.tcx, ty)
                            .map(|(_, def_id)| self.tcx.def_span(def_id));
                        Some((ty.to_string(), span))
                    }
                }
            }
            _ => None,
        }
    }

    /// If the `Self` type of the unsatisfied trait `trait_ref` implements a trait
    /// with the same path as `trait_ref`, a help message about
    /// a probable version mismatch is added to `err`
    fn note_version_mismatch(
        &self,
        err: &mut DiagnosticBuilder<'_>,
        trait_ref: &ty::PolyTraitRef<'tcx>,
    ) {
        let get_trait_impl = |trait_def_id| {
            self.tcx.find_map_relevant_impl(trait_def_id, trait_ref.skip_binder().self_ty(), Some)
        };
        let required_trait_path = self.tcx.def_path_str(trait_ref.def_id());
        let all_traits = self.tcx.all_traits(());
        let traits_with_same_path: std::collections::BTreeSet<_> = all_traits
            .iter()
            .filter(|trait_def_id| **trait_def_id != trait_ref.def_id())
            .filter(|trait_def_id| self.tcx.def_path_str(**trait_def_id) == required_trait_path)
            .collect();
        for trait_with_same_path in traits_with_same_path {
            if let Some(impl_def_id) = get_trait_impl(*trait_with_same_path) {
                let impl_span = self.tcx.def_span(impl_def_id);
                err.span_help(impl_span, "trait impl with same name found");
                let trait_crate = self.tcx.crate_name(trait_with_same_path.krate);
                let crate_msg = format!(
                    "perhaps two different versions of crate `{}` are being used?",
                    trait_crate
                );
                err.note(&crate_msg);
            }
        }
    }

    fn mk_trait_obligation_with_new_self_ty(
        &self,
        param_env: ty::ParamEnv<'tcx>,
        trait_ref: ty::PolyTraitRef<'tcx>,
        new_self_ty: Ty<'tcx>,
    ) -> PredicateObligation<'tcx> {
        assert!(!new_self_ty.has_escaping_bound_vars());

        let trait_ref = trait_ref.map_bound_ref(|tr| ty::TraitRef {
            substs: self.tcx.mk_substs_trait(new_self_ty, &tr.substs[1..]),
            ..*tr
        });

        Obligation::new(
            ObligationCause::dummy(),
            param_env,
            trait_ref.without_const().to_predicate(self.tcx),
        )
    }

    #[instrument(skip(self), level = "debug")]
    fn maybe_report_ambiguity(
        &self,
        obligation: &PredicateObligation<'tcx>,
        body_id: Option<hir::BodyId>,
    ) {
        // Unable to successfully determine, probably means
        // insufficient type information, but could mean
        // ambiguous impls. The latter *ought* to be a
        // coherence violation, so we don't report it here.

        let predicate = self.resolve_vars_if_possible(obligation.predicate);
        let span = obligation.cause.span;

        debug!(
            ?predicate, ?obligation.cause.code,
        );

        // Ambiguity errors are often caused as fallout from earlier
        // errors. So just ignore them if this infcx is tainted.
        if self.is_tainted_by_errors() {
            return;
        }

        let bound_predicate = predicate.kind();
        let mut err = match bound_predicate.skip_binder() {
            ty::PredicateKind::Trait(data) => {
                let trait_ref = bound_predicate.rebind(data.trait_ref);
                debug!(?trait_ref);

                if predicate.references_error() {
                    return;
                }
                // Typically, this ambiguity should only happen if
                // there are unresolved type inference variables
                // (otherwise it would suggest a coherence
                // failure). But given #21974 that is not necessarily
                // the case -- we can have multiple where clauses that
                // are only distinguished by a region, which results
                // in an ambiguity even when all types are fully
                // known, since we don't dispatch based on region
                // relationships.

                // Pick the first substitution that still contains inference variables as the one
                // we're going to emit an error for. If there are none (see above), fall back to
                // the substitution for `Self`.
                let subst = {
                    let substs = data.trait_ref.substs;
                    substs
                        .iter()
                        .find(|s| s.has_infer_types_or_consts())
                        .unwrap_or_else(|| substs[0])
                };

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
                // inhabited. We used to check for `self.tcx.sess.has_errors()` to
                // avoid inundating the user with unnecessary errors, but we now
                // check upstream for type errors and don't add the obligations to
                // begin with in those cases.
                if self.tcx.lang_items().sized_trait() == Some(trait_ref.def_id()) {
                    self.emit_inference_failure_err(body_id, span, subst, vec![], ErrorCode::E0282)
                        .emit();
                    return;
                }
                let impl_candidates = self.find_similar_impl_candidates(trait_ref);
                let mut err = self.emit_inference_failure_err(
                    body_id,
                    span,
                    subst,
                    impl_candidates,
                    ErrorCode::E0283,
                );
                err.note(&format!("cannot satisfy `{}`", predicate));
                if let ObligationCauseCode::ItemObligation(def_id) = obligation.cause.code {
                    self.suggest_fully_qualified_path(&mut err, def_id, span, trait_ref.def_id());
                } else if let (
                    Ok(ref snippet),
                    ObligationCauseCode::BindingObligation(ref def_id, _),
                ) =
                    (self.tcx.sess.source_map().span_to_snippet(span), &obligation.cause.code)
                {
                    let generics = self.tcx.generics_of(*def_id);
                    if generics.params.iter().any(|p| p.name != kw::SelfUpper)
                        && !snippet.ends_with('>')
                        && !generics.has_impl_trait()
                        && !self.tcx.fn_trait_kind_from_lang_item(*def_id).is_some()
                    {
                        // FIXME: To avoid spurious suggestions in functions where type arguments
                        // where already supplied, we check the snippet to make sure it doesn't
                        // end with a turbofish. Ideally we would have access to a `PathSegment`
                        // instead. Otherwise we would produce the following output:
                        //
                        // error[E0283]: type annotations needed
                        //   --> $DIR/issue-54954.rs:3:24
                        //    |
                        // LL | const ARR_LEN: usize = Tt::const_val::<[i8; 123]>();
                        //    |                        ^^^^^^^^^^^^^^^^^^^^^^^^^^
                        //    |                        |
                        //    |                        cannot infer type
                        //    |                        help: consider specifying the type argument
                        //    |                        in the function call:
                        //    |                        `Tt::const_val::<[i8; 123]>::<T>`
                        // ...
                        // LL |     const fn const_val<T: Sized>() -> usize {
                        //    |                        - required by this bound in `Tt::const_val`
                        //    |
                        //    = note: cannot satisfy `_: Tt`

                        err.span_suggestion_verbose(
                            span.shrink_to_hi(),
                            &format!(
                                "consider specifying the type argument{} in the function call",
                                pluralize!(generics.params.len()),
                            ),
                            format!(
                                "::<{}>",
                                generics
                                    .params
                                    .iter()
                                    .map(|p| p.name.to_string())
                                    .collect::<Vec<String>>()
                                    .join(", ")
                            ),
                            Applicability::HasPlaceholders,
                        );
                    }
                }
                err
            }

            ty::PredicateKind::WellFormed(arg) => {
                // Same hacky approach as above to avoid deluging user
                // with error messages.
                if arg.references_error() || self.tcx.sess.has_errors() {
                    return;
                }

                self.emit_inference_failure_err(body_id, span, arg, vec![], ErrorCode::E0282)
            }

            ty::PredicateKind::Subtype(data) => {
                if data.references_error() || self.tcx.sess.has_errors() {
                    // no need to overload user in such cases
                    return;
                }
                let SubtypePredicate { a_is_expected: _, a, b } = data;
                // both must be type variables, or the other would've been instantiated
                assert!(a.is_ty_var() && b.is_ty_var());
                self.emit_inference_failure_err(body_id, span, a.into(), vec![], ErrorCode::E0282)
            }
            ty::PredicateKind::Projection(data) => {
                let self_ty = data.projection_ty.self_ty();
                let ty = data.ty;
                if predicate.references_error() {
                    return;
                }
                if self_ty.needs_infer() && ty.needs_infer() {
                    // We do this for the `foo.collect()?` case to produce a suggestion.
                    let mut err = self.emit_inference_failure_err(
                        body_id,
                        span,
                        self_ty.into(),
                        vec![],
                        ErrorCode::E0284,
                    );
                    err.note(&format!("cannot satisfy `{}`", predicate));
                    err
                } else {
                    let mut err = struct_span_err!(
                        self.tcx.sess,
                        span,
                        E0284,
                        "type annotations needed: cannot satisfy `{}`",
                        predicate,
                    );
                    err.span_label(span, &format!("cannot satisfy `{}`", predicate));
                    err
                }
            }

            _ => {
                if self.tcx.sess.has_errors() {
                    return;
                }
                let mut err = struct_span_err!(
                    self.tcx.sess,
                    span,
                    E0284,
                    "type annotations needed: cannot satisfy `{}`",
                    predicate,
                );
                err.span_label(span, &format!("cannot satisfy `{}`", predicate));
                err
            }
        };
        self.note_obligation_cause(&mut err, obligation);
        err.emit();
    }

    /// Returns `true` if the trait predicate may apply for *some* assignment
    /// to the type parameters.
    fn predicate_can_apply(
        &self,
        param_env: ty::ParamEnv<'tcx>,
        pred: ty::PolyTraitRef<'tcx>,
    ) -> bool {
        struct ParamToVarFolder<'a, 'tcx> {
            infcx: &'a InferCtxt<'a, 'tcx>,
            var_map: FxHashMap<Ty<'tcx>, Ty<'tcx>>,
        }

        impl<'a, 'tcx> TypeFolder<'tcx> for ParamToVarFolder<'a, 'tcx> {
            fn tcx<'b>(&'b self) -> TyCtxt<'tcx> {
                self.infcx.tcx
            }

            fn fold_ty(&mut self, ty: Ty<'tcx>) -> Ty<'tcx> {
                if let ty::Param(ty::ParamTy { name, .. }) = *ty.kind() {
                    let infcx = self.infcx;
                    self.var_map.entry(ty).or_insert_with(|| {
                        infcx.next_ty_var(TypeVariableOrigin {
                            kind: TypeVariableOriginKind::TypeParameterDefinition(name, None),
                            span: DUMMY_SP,
                        })
                    })
                } else {
                    ty.super_fold_with(self)
                }
            }
        }

        self.probe(|_| {
            let mut selcx = SelectionContext::new(self);

            let cleaned_pred =
                pred.fold_with(&mut ParamToVarFolder { infcx: self, var_map: Default::default() });

            let cleaned_pred = super::project::normalize(
                &mut selcx,
                param_env,
                ObligationCause::dummy(),
                cleaned_pred,
            )
            .value;

            let obligation = Obligation::new(
                ObligationCause::dummy(),
                param_env,
                cleaned_pred.without_const().to_predicate(selcx.tcx()),
            );

            self.predicate_may_hold(&obligation)
        })
    }

    fn note_obligation_cause(
        &self,
        err: &mut DiagnosticBuilder<'tcx>,
        obligation: &PredicateObligation<'tcx>,
    ) {
        // First, attempt to add note to this error with an async-await-specific
        // message, and fall back to regular note otherwise.
        if !self.maybe_note_obligation_cause_for_async_await(err, obligation) {
            self.note_obligation_cause_code(
                err,
                &obligation.predicate,
                &obligation.cause.code,
                &mut vec![],
                &mut Default::default(),
            );
            self.suggest_unsized_bound_if_applicable(err, obligation);
        }
    }

    fn suggest_unsized_bound_if_applicable(
        &self,
        err: &mut DiagnosticBuilder<'tcx>,
        obligation: &PredicateObligation<'tcx>,
    ) {
        let (pred, item_def_id, span) =
            match (obligation.predicate.kind().skip_binder(), obligation.cause.code.peel_derives())
            {
                (
                    ty::PredicateKind::Trait(pred),
                    &ObligationCauseCode::BindingObligation(item_def_id, span),
                ) => (pred, item_def_id, span),
                _ => return,
            };
        debug!(
            "suggest_unsized_bound_if_applicable: pred={:?} item_def_id={:?} span={:?}",
            pred, item_def_id, span
        );
        let node = match (
            self.tcx.hir().get_if_local(item_def_id),
            Some(pred.def_id()) == self.tcx.lang_items().sized_trait(),
        ) {
            (Some(node), true) => node,
            _ => return,
        };
        self.maybe_suggest_unsized_generics(err, span, node);
    }

    fn maybe_suggest_unsized_generics(
        &self,
        err: &mut DiagnosticBuilder<'tcx>,
        span: Span,
        node: Node<'hir>,
    ) {
        let generics = match node.generics() {
            Some(generics) => generics,
            None => return,
        };
        let sized_trait = self.tcx.lang_items().sized_trait();
        debug!("maybe_suggest_unsized_generics: generics.params={:?}", generics.params);
        debug!("maybe_suggest_unsized_generics: generics.where_clause={:?}", generics.where_clause);
        let param = generics
            .params
            .iter()
            .filter(|param| param.span == span)
            .filter(|param| {
                // Check that none of the explicit trait bounds is `Sized`. Assume that an explicit
                // `Sized` bound is there intentionally and we don't need to suggest relaxing it.
                param
                    .bounds
                    .iter()
                    .all(|bound| bound.trait_ref().and_then(|tr| tr.trait_def_id()) != sized_trait)
            })
            .next();
        let param = match param {
            Some(param) => param,
            _ => return,
        };
        debug!("maybe_suggest_unsized_generics: param={:?}", param);
        match node {
            hir::Node::Item(
                item
                @
                hir::Item {
                    // Only suggest indirection for uses of type parameters in ADTs.
                    kind:
                        hir::ItemKind::Enum(..) | hir::ItemKind::Struct(..) | hir::ItemKind::Union(..),
                    ..
                },
            ) => {
                if self.maybe_indirection_for_unsized(err, item, param) {
                    return;
                }
            }
            _ => {}
        };
        // Didn't add an indirection suggestion, so add a general suggestion to relax `Sized`.
        let (span, separator) = match param.bounds {
            [] => (span.shrink_to_hi(), ":"),
            [.., bound] => (bound.span().shrink_to_hi(), " +"),
        };
        err.span_suggestion_verbose(
            span,
            "consider relaxing the implicit `Sized` restriction",
            format!("{} ?Sized", separator),
            Applicability::MachineApplicable,
        );
    }

    fn maybe_indirection_for_unsized(
        &self,
        err: &mut DiagnosticBuilder<'tcx>,
        item: &'hir Item<'hir>,
        param: &'hir GenericParam<'hir>,
    ) -> bool {
        // Suggesting `T: ?Sized` is only valid in an ADT if `T` is only used in a
        // borrow. `struct S<'a, T: ?Sized>(&'a T);` is valid, `struct S<T: ?Sized>(T);`
        // is not. Look for invalid "bare" parameter uses, and suggest using indirection.
        let mut visitor =
            FindTypeParam { param: param.name.ident().name, invalid_spans: vec![], nested: false };
        visitor.visit_item(item);
        if visitor.invalid_spans.is_empty() {
            return false;
        }
        let mut multispan: MultiSpan = param.span.into();
        multispan.push_span_label(
            param.span,
            format!("this could be changed to `{}: ?Sized`...", param.name.ident()),
        );
        for sp in visitor.invalid_spans {
            multispan.push_span_label(
                sp,
                format!("...if indirection were used here: `Box<{}>`", param.name.ident()),
            );
        }
        err.span_help(
            multispan,
            &format!(
                "you could relax the implicit `Sized` bound on `{T}` if it were \
                used through indirection like `&{T}` or `Box<{T}>`",
                T = param.name.ident(),
            ),
        );
        true
    }

    fn is_recursive_obligation(
        &self,
        obligated_types: &mut Vec<&ty::TyS<'tcx>>,
        cause_code: &ObligationCauseCode<'tcx>,
    ) -> bool {
        if let ObligationCauseCode::BuiltinDerivedObligation(ref data) = cause_code {
            let parent_trait_ref = self.resolve_vars_if_possible(data.parent_trait_ref);

            if obligated_types.iter().any(|ot| ot == &parent_trait_ref.skip_binder().self_ty()) {
                return true;
            }
        }
        false
    }
}

/// Look for type `param` in an ADT being used only through a reference to confirm that suggesting
/// `param: ?Sized` would be a valid constraint.
struct FindTypeParam {
    param: rustc_span::Symbol,
    invalid_spans: Vec<Span>,
    nested: bool,
}

impl<'v> Visitor<'v> for FindTypeParam {
    type Map = rustc_hir::intravisit::ErasedMap<'v>;

    fn nested_visit_map(&mut self) -> hir::intravisit::NestedVisitorMap<Self::Map> {
        hir::intravisit::NestedVisitorMap::None
    }

    fn visit_where_predicate(&mut self, _: &'v hir::WherePredicate<'v>) {
        // Skip where-clauses, to avoid suggesting indirection for type parameters found there.
    }

    fn visit_ty(&mut self, ty: &hir::Ty<'_>) {
        // We collect the spans of all uses of the "bare" type param, like in `field: T` or
        // `field: (T, T)` where we could make `T: ?Sized` while skipping cases that are known to be
        // valid like `field: &'a T` or `field: *mut T` and cases that *might* have further `Sized`
        // obligations like `Box<T>` and `Vec<T>`, but we perform no extra analysis for those cases
        // and suggest `T: ?Sized` regardless of their obligations. This is fine because the errors
        // in that case should make what happened clear enough.
        match ty.kind {
            hir::TyKind::Ptr(_) | hir::TyKind::Rptr(..) | hir::TyKind::TraitObject(..) => {}
            hir::TyKind::Path(hir::QPath::Resolved(None, path))
                if path.segments.len() == 1 && path.segments[0].ident.name == self.param =>
            {
                if !self.nested {
                    debug!("FindTypeParam::visit_ty: ty={:?}", ty);
                    self.invalid_spans.push(ty.span);
                }
            }
            hir::TyKind::Path(_) => {
                let prev = self.nested;
                self.nested = true;
                hir::intravisit::walk_ty(self, ty);
                self.nested = prev;
            }
            _ => {
                hir::intravisit::walk_ty(self, ty);
            }
        }
    }
}

pub fn recursive_type_with_infinite_size_error(
    tcx: TyCtxt<'tcx>,
    type_def_id: DefId,
    spans: Vec<Span>,
) {
    assert!(type_def_id.is_local());
    let span = tcx.hir().span_if_local(type_def_id).unwrap();
    let span = tcx.sess.source_map().guess_head_span(span);
    let path = tcx.def_path_str(type_def_id);
    let mut err =
        struct_span_err!(tcx.sess, span, E0072, "recursive type `{}` has infinite size", path);
    err.span_label(span, "recursive type has infinite size");
    for &span in &spans {
        err.span_label(span, "recursive without indirection");
    }
    let msg = format!(
        "insert some indirection (e.g., a `Box`, `Rc`, or `&`) to make `{}` representable",
        path,
    );
    if spans.len() <= 4 {
        err.multipart_suggestion(
            &msg,
            spans
                .iter()
                .flat_map(|&span| {
                    vec![
                        (span.shrink_to_lo(), "Box<".to_string()),
                        (span.shrink_to_hi(), ">".to_string()),
                    ]
                    .into_iter()
                })
                .collect(),
            Applicability::HasPlaceholders,
        );
    } else {
        err.help(&msg);
    }
    err.emit();
}

/// Summarizes information
#[derive(Clone)]
pub enum ArgKind {
    /// An argument of non-tuple type. Parameters are (name, ty)
    Arg(String, String),

    /// An argument of tuple type. For a "found" argument, the span is
    /// the location in the source of the pattern. For an "expected"
    /// argument, it will be None. The vector is a list of (name, ty)
    /// strings for the components of the tuple.
    Tuple(Option<Span>, Vec<(String, String)>),
}

impl ArgKind {
    fn empty() -> ArgKind {
        ArgKind::Arg("_".to_owned(), "_".to_owned())
    }

    /// Creates an `ArgKind` from the expected type of an
    /// argument. It has no name (`_`) and an optional source span.
    pub fn from_expected_ty(t: Ty<'_>, span: Option<Span>) -> ArgKind {
        match t.kind() {
            ty::Tuple(tys) => ArgKind::Tuple(
                span,
                tys.iter().map(|ty| ("_".to_owned(), ty.to_string())).collect::<Vec<_>>(),
            ),
            _ => ArgKind::Arg("_".to_owned(), t.to_string()),
        }
    }
}
