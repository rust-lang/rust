use core::ops::ControlFlow;
use std::borrow::Cow;
use std::path::PathBuf;

use rustc_abi::ExternAbi;
use rustc_ast::TraitObjectSyntax;
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::unord::UnordSet;
use rustc_errors::codes::*;
use rustc_errors::{
    Applicability, Diag, ErrorGuaranteed, Level, MultiSpan, StashKey, StringPart, Suggestions,
    pluralize, struct_span_code_err,
};
use rustc_hir::def_id::{DefId, LOCAL_CRATE, LocalDefId};
use rustc_hir::intravisit::Visitor;
use rustc_hir::{self as hir, LangItem, Node};
use rustc_infer::infer::{InferOk, TypeTrace};
use rustc_infer::traits::ImplSource;
use rustc_infer::traits::solve::Goal;
use rustc_middle::traits::SignatureMismatchData;
use rustc_middle::traits::select::OverflowError;
use rustc_middle::ty::abstract_const::NotConstEvaluatable;
use rustc_middle::ty::error::{ExpectedFound, TypeError};
use rustc_middle::ty::print::{
    PrintPolyTraitPredicateExt, PrintTraitPredicateExt as _, PrintTraitRefExt as _,
    with_forced_trimmed_paths,
};
use rustc_middle::ty::{
    self, TraitRef, Ty, TyCtxt, TypeFoldable, TypeFolder, TypeSuperFoldable, TypeVisitableExt,
    Upcast,
};
use rustc_middle::{bug, span_bug};
use rustc_span::{BytePos, DUMMY_SP, STDLIB_STABLE_CRATES, Span, Symbol, sym};
use tracing::{debug, instrument};

use super::on_unimplemented::{AppendConstMessage, OnUnimplementedNote};
use super::suggestions::get_explanation_based_on_obligation;
use super::{
    ArgKind, CandidateSimilarity, FindExprBySpan, GetSafeTransmuteErrorAndReason, ImplCandidate,
    UnsatisfiedConst,
};
use crate::error_reporting::TypeErrCtxt;
use crate::error_reporting::infer::TyCategory;
use crate::error_reporting::traits::report_dyn_incompatibility;
use crate::errors::{ClosureFnMutLabel, ClosureFnOnceLabel, ClosureKindMismatch, CoroClosureNotFn};
use crate::infer::{self, InferCtxt, InferCtxtExt as _};
use crate::traits::query::evaluate_obligation::InferCtxtExt as _;
use crate::traits::{
    MismatchedProjectionTypes, NormalizeExt, Obligation, ObligationCause, ObligationCauseCode,
    ObligationCtxt, Overflow, PredicateObligation, SelectionContext, SelectionError,
    SignatureMismatch, TraitDynIncompatible, elaborate, specialization_graph,
};

impl<'a, 'tcx> TypeErrCtxt<'a, 'tcx> {
    /// The `root_obligation` parameter should be the `root_obligation` field
    /// from a `FulfillmentError`. If no `FulfillmentError` is available,
    /// then it should be the same as `obligation`.
    pub fn report_selection_error(
        &self,
        mut obligation: PredicateObligation<'tcx>,
        root_obligation: &PredicateObligation<'tcx>,
        error: &SelectionError<'tcx>,
    ) -> ErrorGuaranteed {
        let tcx = self.tcx;
        let mut span = obligation.cause.span;
        let mut long_ty_file = None;

        let mut err = match *error {
            SelectionError::Unimplemented => {
                // If this obligation was generated as a result of well-formedness checking, see if we
                // can get a better error message by performing HIR-based well-formedness checking.
                if let ObligationCauseCode::WellFormed(Some(wf_loc)) =
                    root_obligation.cause.code().peel_derives()
                    && !obligation.predicate.has_non_region_infer()
                {
                    if let Some(cause) = self
                        .tcx
                        .diagnostic_hir_wf_check((tcx.erase_regions(obligation.predicate), *wf_loc))
                    {
                        obligation.cause = cause.clone();
                        span = obligation.cause.span;
                    }
                }

                if let ObligationCauseCode::CompareImplItem {
                    impl_item_def_id,
                    trait_item_def_id,
                    kind: _,
                } = *obligation.cause.code()
                {
                    debug!("ObligationCauseCode::CompareImplItemObligation");
                    return self.report_extra_impl_obligation(
                        span,
                        impl_item_def_id,
                        trait_item_def_id,
                        &format!("`{}`", obligation.predicate),
                    )
                    .emit()
                }

                // Report a const-param specific error
                if let ObligationCauseCode::ConstParam(ty) = *obligation.cause.code().peel_derives()
                {
                    return self.report_const_param_not_wf(ty, &obligation).emit();
                }

                let bound_predicate = obligation.predicate.kind();
                match bound_predicate.skip_binder() {
                    ty::PredicateKind::Clause(ty::ClauseKind::Trait(trait_predicate)) => {
                        let leaf_trait_predicate =
                            self.resolve_vars_if_possible(bound_predicate.rebind(trait_predicate));

                        // Let's use the root obligation as the main message, when we care about the
                        // most general case ("X doesn't implement Pattern<'_>") over the case that
                        // happened to fail ("char doesn't implement Fn(&mut char)").
                        //
                        // We rely on a few heuristics to identify cases where this root
                        // obligation is more important than the leaf obligation:
                        let (main_trait_predicate, main_obligation) = if let ty::PredicateKind::Clause(
                            ty::ClauseKind::Trait(root_pred)
                        ) = root_obligation.predicate.kind().skip_binder()
                            && !leaf_trait_predicate.self_ty().skip_binder().has_escaping_bound_vars()
                            && !root_pred.self_ty().has_escaping_bound_vars()
                            // The type of the leaf predicate is (roughly) the same as the type
                            // from the root predicate, as a proxy for "we care about the root"
                            // FIXME: this doesn't account for trivial derefs, but works as a first
                            // approximation.
                            && (
                                // `T: Trait` && `&&T: OtherTrait`, we want `OtherTrait`
                                self.can_eq(
                                    obligation.param_env,
                                    leaf_trait_predicate.self_ty().skip_binder(),
                                    root_pred.self_ty().peel_refs(),
                                )
                                // `&str: Iterator` && `&str: IntoIterator`, we want `IntoIterator`
                                || self.can_eq(
                                    obligation.param_env,
                                    leaf_trait_predicate.self_ty().skip_binder(),
                                    root_pred.self_ty(),
                                )
                            )
                            // The leaf trait and the root trait are different, so as to avoid
                            // talking about `&mut T: Trait` and instead remain talking about
                            // `T: Trait` instead
                            && leaf_trait_predicate.def_id() != root_pred.def_id()
                            // The root trait is not `Unsize`, as to avoid talking about it in
                            // `tests/ui/coercion/coerce-issue-49593-box-never.rs`.
                            && !self.tcx.is_lang_item(root_pred.def_id(), LangItem::Unsize)
                        {
                            (
                                self.resolve_vars_if_possible(
                                    root_obligation.predicate.kind().rebind(root_pred),
                                ),
                                root_obligation,
                            )
                        } else {
                            (leaf_trait_predicate, &obligation)
                        };

                        if let Some(guar) = self.emit_specialized_closure_kind_error(
                            &obligation,
                            leaf_trait_predicate,
                        ) {
                            return guar;
                        }

                        if let Err(guar) = leaf_trait_predicate.error_reported()
                        {
                            return guar;
                        }
                        // Silence redundant errors on binding access that are already
                        // reported on the binding definition (#56607).
                        if let Err(guar) = self.fn_arg_obligation(&obligation) {
                            return guar;
                        }
                        let (post_message, pre_message, type_def) = self
                            .get_parent_trait_ref(obligation.cause.code())
                            .map(|(t, s)| {
                                let t = self.tcx.short_string(t, &mut long_ty_file);
                                (
                                    format!(" in `{t}`"),
                                    format!("within `{t}`, "),
                                    s.map(|s| (format!("within this `{t}`"), s)),
                                )
                            })
                            .unwrap_or_default();

                        let OnUnimplementedNote {
                            message,
                            label,
                            notes,
                            parent_label,
                            append_const_msg,
                        } = self.on_unimplemented_note(main_trait_predicate, main_obligation, &mut long_ty_file);

                        let have_alt_message = message.is_some() || label.is_some();
                        let is_try_conversion = self.is_try_conversion(span, main_trait_predicate.def_id());
                        let is_question_mark = matches!(
                            root_obligation.cause.code().peel_derives(),
                            ObligationCauseCode::QuestionMark,
                        ) && !(
                            self.tcx.is_diagnostic_item(sym::FromResidual, main_trait_predicate.def_id())
                                || self.tcx.is_lang_item(main_trait_predicate.def_id(), LangItem::Try)
                        );
                        let is_unsize =
                            self.tcx.is_lang_item(leaf_trait_predicate.def_id(), LangItem::Unsize);
                        let question_mark_message = "the question mark operation (`?`) implicitly \
                                                     performs a conversion on the error value \
                                                     using the `From` trait";
                        let (message, notes, append_const_msg) = if is_try_conversion {
                            // We have a `-> Result<_, E1>` and `gives_E2()?`.
                            (
                                Some(format!(
                                    "`?` couldn't convert the error to `{}`",
                                    main_trait_predicate.skip_binder().self_ty(),
                                )),
                                vec![question_mark_message.to_owned()],
                                Some(AppendConstMessage::Default),
                            )
                        } else if is_question_mark {
                            // Similar to the case above, but in this case the conversion is for a
                            // trait object: `-> Result<_, Box<dyn Error>` and `gives_E()?` when
                            // `E: Error` isn't met.
                            (
                                Some(format!(
                                    "`?` couldn't convert the error: `{main_trait_predicate}` is \
                                     not satisfied",
                                )),
                                vec![question_mark_message.to_owned()],
                                Some(AppendConstMessage::Default),
                            )
                        } else {
                            (message, notes, append_const_msg)
                        };

                        let err_msg = self.get_standard_error_message(
                            main_trait_predicate,
                            message,
                            None,
                            append_const_msg,
                            post_message,
                            &mut long_ty_file,
                        );

                        let (err_msg, safe_transmute_explanation) = if self.tcx.is_lang_item(
                            main_trait_predicate.def_id(),
                            LangItem::TransmuteTrait,
                        ) {
                            // Recompute the safe transmute reason and use that for the error reporting
                            match self.get_safe_transmute_error_and_reason(
                                obligation.clone(),
                                main_trait_predicate,
                                span,
                            ) {
                                GetSafeTransmuteErrorAndReason::Silent => {
                                    return self.dcx().span_delayed_bug(
                                        span, "silent safe transmute error"
                                    );
                                }
                                GetSafeTransmuteErrorAndReason::Default => {
                                    (err_msg, None)
                                }
                                GetSafeTransmuteErrorAndReason::Error {
                                    err_msg,
                                    safe_transmute_explanation,
                                } => (err_msg, safe_transmute_explanation),
                            }
                        } else {
                            (err_msg, None)
                        };

                        let mut err = struct_span_code_err!(self.dcx(), span, E0277, "{}", err_msg);
                        *err.long_ty_path() = long_ty_file;

                        let mut suggested = false;
                        if is_try_conversion || is_question_mark {
                            suggested = self.try_conversion_context(&obligation, main_trait_predicate, &mut err);
                        }

                        if let Some(ret_span) = self.return_type_span(&obligation) {
                            if is_try_conversion {
                                err.span_label(
                                    ret_span,
                                    format!(
                                        "expected `{}` because of this",
                                        main_trait_predicate.skip_binder().self_ty()
                                    ),
                                );
                            } else if is_question_mark {
                                err.span_label(ret_span, format!("required `{main_trait_predicate}` because of this"));
                            }
                        }

                        if tcx.is_lang_item(leaf_trait_predicate.def_id(), LangItem::Tuple) {
                            self.add_tuple_trait_message(
                                obligation.cause.code().peel_derives(),
                                &mut err,
                            );
                        }

                        let explanation = get_explanation_based_on_obligation(
                            self.tcx,
                            &obligation,
                            leaf_trait_predicate,
                            pre_message,
                        );

                        self.check_for_binding_assigned_block_without_tail_expression(
                            &obligation,
                            &mut err,
                            leaf_trait_predicate,
                        );
                        self.suggest_add_result_as_return_type(
                            &obligation,
                            &mut err,
                            leaf_trait_predicate,
                        );

                        if self.suggest_add_reference_to_arg(
                            &obligation,
                            &mut err,
                            leaf_trait_predicate,
                            have_alt_message,
                        ) {
                            self.note_obligation_cause(&mut err, &obligation);
                            return err.emit();
                        }

                        if let Some(s) = label {
                            // If it has a custom `#[rustc_on_unimplemented]`
                            // error message, let's display it as the label!
                            err.span_label(span, s);
                            if !matches!(leaf_trait_predicate.skip_binder().self_ty().kind(), ty::Param(_))
                                // When the self type is a type param We don't need to "the trait
                                // `std::marker::Sized` is not implemented for `T`" as we will point
                                // at the type param with a label to suggest constraining it.
                                && !self.tcx.is_diagnostic_item(sym::FromResidual, leaf_trait_predicate.def_id())
                                    // Don't say "the trait `FromResidual<Option<Infallible>>` is
                                    // not implemented for `Result<T, E>`".
                            {
                                err.help(explanation);
                            }
                        } else if let Some(custom_explanation) = safe_transmute_explanation {
                            err.span_label(span, custom_explanation);
                        } else if explanation.len() > self.tcx.sess.diagnostic_width() {
                            // Really long types don't look good as span labels, instead move it
                            // to a `help`.
                            err.span_label(span, "unsatisfied trait bound");
                            err.help(explanation);
                        } else {
                            err.span_label(span, explanation);
                        }

                        if let ObligationCauseCode::Coercion { source, target } =
                            *obligation.cause.code().peel_derives()
                        {
                            if self.tcx.is_lang_item(leaf_trait_predicate.def_id(), LangItem::Sized) {
                                self.suggest_borrowing_for_object_cast(
                                    &mut err,
                                    root_obligation,
                                    source,
                                    target,
                                );
                            }
                        }

                        let UnsatisfiedConst(unsatisfied_const) = self
                            .maybe_add_note_for_unsatisfied_const(
                                leaf_trait_predicate,
                                &mut err,
                                span,
                            );

                        if let Some((msg, span)) = type_def {
                            err.span_label(span, msg);
                        }
                        for note in notes {
                            // If it has a custom `#[rustc_on_unimplemented]` note, let's display it
                            err.note(note);
                        }
                        if let Some(s) = parent_label {
                            let body = obligation.cause.body_id;
                            err.span_label(tcx.def_span(body), s);
                        }

                        self.suggest_floating_point_literal(&obligation, &mut err, leaf_trait_predicate);
                        self.suggest_dereferencing_index(&obligation, &mut err, leaf_trait_predicate);
                        suggested |= self.suggest_dereferences(&obligation, &mut err, leaf_trait_predicate);
                        suggested |= self.suggest_fn_call(&obligation, &mut err, leaf_trait_predicate);
                        let impl_candidates = self.find_similar_impl_candidates(leaf_trait_predicate);
                        suggested = if let &[cand] = &impl_candidates[..] {
                            let cand = cand.trait_ref;
                            if let (ty::FnPtr(..), ty::FnDef(..)) =
                                (cand.self_ty().kind(), main_trait_predicate.self_ty().skip_binder().kind())
                            {
                                // Wrap method receivers and `&`-references in parens
                                let suggestion = if self.tcx.sess.source_map().span_look_ahead(span, ".", Some(50)).is_some() {
                                    vec![
                                        (span.shrink_to_lo(), format!("(")),
                                        (span.shrink_to_hi(), format!(" as {})", cand.self_ty())),
                                    ]
                                } else if let Some(body) = self.tcx.hir_maybe_body_owned_by(obligation.cause.body_id) {
                                    let mut expr_finder = FindExprBySpan::new(span, self.tcx);
                                    expr_finder.visit_expr(body.value);
                                    if let Some(expr) = expr_finder.result &&
                                        let hir::ExprKind::AddrOf(_, _, expr) = expr.kind {
                                        vec![
                                            (expr.span.shrink_to_lo(), format!("(")),
                                            (expr.span.shrink_to_hi(), format!(" as {})", cand.self_ty())),
                                        ]
                                    } else {
                                        vec![(span.shrink_to_hi(), format!(" as {}", cand.self_ty()))]
                                    }
                                } else {
                                    vec![(span.shrink_to_hi(), format!(" as {}", cand.self_ty()))]
                                };
                                err.multipart_suggestion(
                                    format!(
                                        "the trait `{}` is implemented for fn pointer `{}`, try casting using `as`",
                                        cand.print_trait_sugared(),
                                        cand.self_ty(),
                                    ),
                                    suggestion,
                                    Applicability::MaybeIncorrect,
                                );
                                true
                            } else {
                                false
                            }
                        } else {
                            false
                        } || suggested;
                        suggested |=
                            self.suggest_remove_reference(&obligation, &mut err, leaf_trait_predicate);
                        suggested |= self.suggest_semicolon_removal(
                            &obligation,
                            &mut err,
                            span,
                            leaf_trait_predicate,
                        );
                        self.note_version_mismatch(&mut err, leaf_trait_predicate);
                        self.suggest_remove_await(&obligation, &mut err);
                        self.suggest_derive(&obligation, &mut err, leaf_trait_predicate);

                        if tcx.is_lang_item(leaf_trait_predicate.def_id(), LangItem::Try) {
                            self.suggest_await_before_try(
                                &mut err,
                                &obligation,
                                leaf_trait_predicate,
                                span,
                            );
                        }

                        if self.suggest_add_clone_to_arg(&obligation, &mut err, leaf_trait_predicate) {
                            return err.emit();
                        }

                        if self.suggest_impl_trait(&mut err, &obligation, leaf_trait_predicate) {
                            return err.emit();
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

                        let is_fn_trait = tcx.is_fn_trait(leaf_trait_predicate.def_id());
                        let is_target_feature_fn = if let ty::FnDef(def_id, _) =
                            *leaf_trait_predicate.skip_binder().self_ty().kind()
                        {
                            !self.tcx.codegen_fn_attrs(def_id).target_features.is_empty()
                        } else {
                            false
                        };
                        if is_fn_trait && is_target_feature_fn {
                            err.note(
                                "`#[target_feature]` functions do not implement the `Fn` traits",
                            );
                            err.note(
                                "try casting the function to a `fn` pointer or wrapping it in a closure",
                            );
                        }

                        self.try_to_add_help_message(
                            &root_obligation,
                            &obligation,
                            leaf_trait_predicate,
                            &mut err,
                            span,
                            is_fn_trait,
                            suggested,
                            unsatisfied_const,
                        );

                        // Changing mutability doesn't make a difference to whether we have
                        // an `Unsize` impl (Fixes ICE in #71036)
                        if !is_unsize {
                            self.suggest_change_mut(&obligation, &mut err, leaf_trait_predicate);
                        }

                        // If this error is due to `!: Trait` not implemented but `(): Trait` is
                        // implemented, and fallback has occurred, then it could be due to a
                        // variable that used to fallback to `()` now falling back to `!`. Issue a
                        // note informing about the change in behaviour.
                        if leaf_trait_predicate.skip_binder().self_ty().is_never()
                            && self.fallback_has_occurred
                        {
                            let predicate = leaf_trait_predicate.map_bound(|trait_pred| {
                                trait_pred.with_self_ty(self.tcx, tcx.types.unit)
                            });
                            let unit_obligation = obligation.with(tcx, predicate);
                            if self.predicate_may_hold(&unit_obligation) {
                                err.note(
                                    "this error might have been caused by changes to \
                                    Rust's type-inference algorithm (see issue #48950 \
                                    <https://github.com/rust-lang/rust/issues/48950> \
                                    for more information)",
                                );
                                err.help("did you intend to use the type `()` here instead?");
                            }
                        }

                        self.explain_hrtb_projection(&mut err, leaf_trait_predicate, obligation.param_env, &obligation.cause);
                        self.suggest_desugaring_async_fn_in_trait(&mut err, main_trait_predicate);

                        // Return early if the trait is Debug or Display and the invocation
                        // originates within a standard library macro, because the output
                        // is otherwise overwhelming and unhelpful (see #85844 for an
                        // example).

                        let in_std_macro =
                            match obligation.cause.span.ctxt().outer_expn_data().macro_def_id {
                                Some(macro_def_id) => {
                                    let crate_name = tcx.crate_name(macro_def_id.krate);
                                    STDLIB_STABLE_CRATES.contains(&crate_name)
                                }
                                None => false,
                            };

                        if in_std_macro
                            && matches!(
                                self.tcx.get_diagnostic_name(leaf_trait_predicate.def_id()),
                                Some(sym::Debug | sym::Display)
                            )
                        {
                            return err.emit();
                        }

                        err
                    }

                    ty::PredicateKind::Clause(ty::ClauseKind::HostEffect(predicate)) => {
                        self.report_host_effect_error(bound_predicate.rebind(predicate), obligation.param_env, span)
                    }

                    ty::PredicateKind::Subtype(predicate) => {
                        // Errors for Subtype predicates show up as
                        // `FulfillmentErrorCode::SubtypeError`,
                        // not selection error.
                        span_bug!(span, "subtype requirement gave wrong error: `{:?}`", predicate)
                    }

                    ty::PredicateKind::Coerce(predicate) => {
                        // Errors for Coerce predicates show up as
                        // `FulfillmentErrorCode::SubtypeError`,
                        // not selection error.
                        span_bug!(span, "coerce requirement gave wrong error: `{:?}`", predicate)
                    }

                    ty::PredicateKind::Clause(ty::ClauseKind::RegionOutlives(..))
                    | ty::PredicateKind::Clause(ty::ClauseKind::TypeOutlives(..)) => {
                        span_bug!(
                            span,
                            "outlives clauses should not error outside borrowck. obligation: `{:?}`",
                            obligation
                        )
                    }

                    ty::PredicateKind::Clause(ty::ClauseKind::Projection(..)) => {
                        span_bug!(
                            span,
                            "projection clauses should be implied from elsewhere. obligation: `{:?}`",
                            obligation
                        )
                    }

                    ty::PredicateKind::DynCompatible(trait_def_id) => {
                        let violations = self.tcx.dyn_compatibility_violations(trait_def_id);
                        let mut err = report_dyn_incompatibility(
                            self.tcx,
                            span,
                            None,
                            trait_def_id,
                            violations,
                        );
                        if let hir::Node::Item(item) =
                            self.tcx.hir_node_by_def_id(obligation.cause.body_id)
                            && let hir::ItemKind::Impl(impl_) = item.kind
                            && let None = impl_.of_trait
                            && let hir::TyKind::TraitObject(_, tagged_ptr) = impl_.self_ty.kind
                            && let TraitObjectSyntax::None = tagged_ptr.tag()
                            && impl_.self_ty.span.edition().at_least_rust_2021()
                        {
                            // Silence the dyn-compatibility error in favor of the missing dyn on
                            // self type error. #131051.
                            err.downgrade_to_delayed_bug();
                        }
                        err
                    }

                    ty::PredicateKind::Clause(ty::ClauseKind::WellFormed(ty)) => {
                        let ty = self.resolve_vars_if_possible(ty);
                        if self.next_trait_solver() {
                            if let Err(guar) = ty.error_reported() {
                                return guar;
                            }

                            // FIXME: we'll need a better message which takes into account
                            // which bounds actually failed to hold.
                            self.dcx().struct_span_err(
                                span,
                                format!("the type `{ty}` is not well-formed"),
                            )
                        } else {
                            // WF predicates cannot themselves make
                            // errors. They can only block due to
                            // ambiguity; otherwise, they always
                            // degenerate into other obligations
                            // (which may fail).
                            span_bug!(span, "WF predicate not satisfied for {:?}", ty);
                        }
                    }

                    // Errors for `ConstEvaluatable` predicates show up as
                    // `SelectionError::ConstEvalFailure`,
                    // not `Unimplemented`.
                    ty::PredicateKind::Clause(ty::ClauseKind::ConstEvaluatable(..))
                    // Errors for `ConstEquate` predicates show up as
                    // `SelectionError::ConstEvalFailure`,
                    // not `Unimplemented`.
                    | ty::PredicateKind::ConstEquate { .. }
                    // Ambiguous predicates should never error
                    | ty::PredicateKind::Ambiguous
                    | ty::PredicateKind::NormalizesTo { .. }
                    | ty::PredicateKind::AliasRelate { .. }
                    | ty::PredicateKind::Clause(ty::ClauseKind::ConstArgHasType { .. }) => {
                        span_bug!(
                            span,
                            "Unexpected `Predicate` for `SelectionError`: `{:?}`",
                            obligation
                        )
                    }
                }
            }

            SignatureMismatch(box SignatureMismatchData {
                found_trait_ref,
                expected_trait_ref,
                terr: terr @ TypeError::CyclicTy(_),
            }) => self.report_cyclic_signature_error(
                &obligation,
                found_trait_ref,
                expected_trait_ref,
                terr,
            ),
            SignatureMismatch(box SignatureMismatchData {
                found_trait_ref,
                expected_trait_ref,
                terr: _,
            }) => {
                match self.report_signature_mismatch_error(
                    &obligation,
                    span,
                    found_trait_ref,
                    expected_trait_ref,
                ) {
                    Ok(err) => err,
                    Err(guar) => return guar,
                }
            }

            SelectionError::OpaqueTypeAutoTraitLeakageUnknown(def_id) => return self.report_opaque_type_auto_trait_leakage(
                &obligation,
                def_id,
            ),

            TraitDynIncompatible(did) => {
                let violations = self.tcx.dyn_compatibility_violations(did);
                report_dyn_incompatibility(self.tcx, span, None, did, violations)
            }

            SelectionError::NotConstEvaluatable(NotConstEvaluatable::MentionsInfer) => {
                bug!(
                    "MentionsInfer should have been handled in `traits/fulfill.rs` or `traits/select/mod.rs`"
                )
            }
            SelectionError::NotConstEvaluatable(NotConstEvaluatable::MentionsParam) => {
                match self.report_not_const_evaluatable_error(&obligation, span) {
                    Ok(err) => err,
                    Err(guar) => return guar,
                }
            }

            // Already reported in the query.
            SelectionError::NotConstEvaluatable(NotConstEvaluatable::Error(guar)) |
            // Already reported.
            Overflow(OverflowError::Error(guar)) => {
                self.set_tainted_by_errors(guar);
                return guar
            },

            Overflow(_) => {
                bug!("overflow should be handled before the `report_selection_error` path");
            }

            SelectionError::ConstArgHasWrongType { ct, ct_ty, expected_ty } => {
                let mut diag = self.dcx().struct_span_err(
                    span,
                    format!("the constant `{ct}` is not of type `{expected_ty}`"),
                );

                self.note_type_err(
                    &mut diag,
                    &obligation.cause,
                    None,
                    None,
                    TypeError::Sorts(ty::error::ExpectedFound::new(expected_ty, ct_ty)),
                    false,
                    None,
                );
                diag
            }
        };

        self.note_obligation_cause(&mut err, &obligation);
        err.emit()
    }
}

impl<'a, 'tcx> TypeErrCtxt<'a, 'tcx> {
    pub(super) fn apply_do_not_recommend(
        &self,
        obligation: &mut PredicateObligation<'tcx>,
    ) -> bool {
        let mut base_cause = obligation.cause.code().clone();
        let mut applied_do_not_recommend = false;
        loop {
            if let ObligationCauseCode::ImplDerived(ref c) = base_cause {
                if self.tcx.do_not_recommend_impl(c.impl_or_alias_def_id) {
                    let code = (*c.derived.parent_code).clone();
                    obligation.cause.map_code(|_| code);
                    obligation.predicate = c.derived.parent_trait_pred.upcast(self.tcx);
                    applied_do_not_recommend = true;
                }
            }
            if let Some(parent_cause) = base_cause.parent() {
                base_cause = parent_cause.clone();
            } else {
                break;
            }
        }

        applied_do_not_recommend
    }

    fn report_host_effect_error(
        &self,
        predicate: ty::Binder<'tcx, ty::HostEffectPredicate<'tcx>>,
        param_env: ty::ParamEnv<'tcx>,
        span: Span,
    ) -> Diag<'a> {
        // FIXME(const_trait_impl): We should recompute the predicate with `~const`
        // if it's `const`, and if it holds, explain that this bound only
        // *conditionally* holds. If that fails, we should also do selection
        // to drill this down to an impl or built-in source, so we can
        // point at it and explain that while the trait *is* implemented,
        // that implementation is not const.
        let trait_ref = predicate.map_bound(|predicate| ty::TraitPredicate {
            trait_ref: predicate.trait_ref,
            polarity: ty::PredicatePolarity::Positive,
        });
        let mut file = None;
        let err_msg = self.get_standard_error_message(
            trait_ref,
            None,
            Some(predicate.constness()),
            None,
            String::new(),
            &mut file,
        );
        let mut diag = struct_span_code_err!(self.dcx(), span, E0277, "{}", err_msg);
        *diag.long_ty_path() = file;
        if !self.predicate_may_hold(&Obligation::new(
            self.tcx,
            ObligationCause::dummy(),
            param_env,
            trait_ref,
        )) {
            diag.downgrade_to_delayed_bug();
        }
        diag
    }

    fn emit_specialized_closure_kind_error(
        &self,
        obligation: &PredicateObligation<'tcx>,
        mut trait_pred: ty::PolyTraitPredicate<'tcx>,
    ) -> Option<ErrorGuaranteed> {
        // If we end up on an `AsyncFnKindHelper` goal, try to unwrap the parent
        // `AsyncFn*` goal.
        if self.tcx.is_lang_item(trait_pred.def_id(), LangItem::AsyncFnKindHelper) {
            let mut code = obligation.cause.code();
            // Unwrap a `FunctionArg` cause, which has been refined from a derived obligation.
            if let ObligationCauseCode::FunctionArg { parent_code, .. } = code {
                code = &**parent_code;
            }
            // If we have a derived obligation, then the parent will be a `AsyncFn*` goal.
            if let Some((_, Some(parent))) = code.parent_with_predicate() {
                trait_pred = parent;
            }
        }

        let self_ty = trait_pred.self_ty().skip_binder();

        let (expected_kind, trait_prefix) =
            if let Some(expected_kind) = self.tcx.fn_trait_kind_from_def_id(trait_pred.def_id()) {
                (expected_kind, "")
            } else if let Some(expected_kind) =
                self.tcx.async_fn_trait_kind_from_def_id(trait_pred.def_id())
            {
                (expected_kind, "Async")
            } else {
                return None;
            };

        let (closure_def_id, found_args, has_self_borrows) = match *self_ty.kind() {
            ty::Closure(def_id, args) => {
                (def_id, args.as_closure().sig().map_bound(|sig| sig.inputs()[0]), false)
            }
            ty::CoroutineClosure(def_id, args) => (
                def_id,
                args.as_coroutine_closure()
                    .coroutine_closure_sig()
                    .map_bound(|sig| sig.tupled_inputs_ty),
                !args.as_coroutine_closure().tupled_upvars_ty().is_ty_var()
                    && args.as_coroutine_closure().has_self_borrows(),
            ),
            _ => return None,
        };

        let expected_args = trait_pred.map_bound(|trait_pred| trait_pred.trait_ref.args.type_at(1));

        // Verify that the arguments are compatible. If the signature is
        // mismatched, then we have a totally different error to report.
        if self.enter_forall(found_args, |found_args| {
            self.enter_forall(expected_args, |expected_args| {
                !self.can_eq(obligation.param_env, expected_args, found_args)
            })
        }) {
            return None;
        }

        if let Some(found_kind) = self.closure_kind(self_ty)
            && !found_kind.extends(expected_kind)
        {
            let mut err = self.report_closure_error(
                &obligation,
                closure_def_id,
                found_kind,
                expected_kind,
                trait_prefix,
            );
            self.note_obligation_cause(&mut err, &obligation);
            return Some(err.emit());
        }

        // If the closure has captures, then perhaps the reason that the trait
        // is unimplemented is because async closures don't implement `Fn`/`FnMut`
        // if they have captures.
        if has_self_borrows && expected_kind != ty::ClosureKind::FnOnce {
            let coro_kind = match self
                .tcx
                .coroutine_kind(self.tcx.coroutine_for_closure(closure_def_id))
                .unwrap()
            {
                rustc_hir::CoroutineKind::Desugared(desugaring, _) => desugaring.to_string(),
                coro => coro.to_string(),
            };
            let mut err = self.dcx().create_err(CoroClosureNotFn {
                span: self.tcx.def_span(closure_def_id),
                kind: expected_kind.as_str(),
                coro_kind,
            });
            self.note_obligation_cause(&mut err, &obligation);
            return Some(err.emit());
        }

        None
    }

    fn fn_arg_obligation(
        &self,
        obligation: &PredicateObligation<'tcx>,
    ) -> Result<(), ErrorGuaranteed> {
        if let ObligationCauseCode::FunctionArg { arg_hir_id, .. } = obligation.cause.code()
            && let Node::Expr(arg) = self.tcx.hir_node(*arg_hir_id)
            && let arg = arg.peel_borrows()
            && let hir::ExprKind::Path(hir::QPath::Resolved(
                None,
                hir::Path { res: hir::def::Res::Local(hir_id), .. },
            )) = arg.kind
            && let Node::Pat(pat) = self.tcx.hir_node(*hir_id)
            && let Some((preds, guar)) = self.reported_trait_errors.borrow().get(&pat.span)
            && preds.contains(&obligation.as_goal())
        {
            return Err(*guar);
        }
        Ok(())
    }

    /// When the `E` of the resulting `Result<T, E>` in an expression `foo().bar().baz()?`,
    /// identify those method chain sub-expressions that could or could not have been annotated
    /// with `?`.
    fn try_conversion_context(
        &self,
        obligation: &PredicateObligation<'tcx>,
        trait_pred: ty::PolyTraitPredicate<'tcx>,
        err: &mut Diag<'_>,
    ) -> bool {
        let span = obligation.cause.span;
        /// Look for the (direct) sub-expr of `?`, and return it if it's a `.` method call.
        struct FindMethodSubexprOfTry {
            search_span: Span,
        }
        impl<'v> Visitor<'v> for FindMethodSubexprOfTry {
            type Result = ControlFlow<&'v hir::Expr<'v>>;
            fn visit_expr(&mut self, ex: &'v hir::Expr<'v>) -> Self::Result {
                if let hir::ExprKind::Match(expr, _arms, hir::MatchSource::TryDesugar(_)) = ex.kind
                    && ex.span.with_lo(ex.span.hi() - BytePos(1)).source_equal(self.search_span)
                    && let hir::ExprKind::Call(_, [expr, ..]) = expr.kind
                {
                    ControlFlow::Break(expr)
                } else {
                    hir::intravisit::walk_expr(self, ex)
                }
            }
        }
        let hir_id = self.tcx.local_def_id_to_hir_id(obligation.cause.body_id);
        let Some(body_id) = self.tcx.hir_node(hir_id).body_id() else { return false };
        let ControlFlow::Break(expr) =
            (FindMethodSubexprOfTry { search_span: span }).visit_body(self.tcx.hir_body(body_id))
        else {
            return false;
        };
        let Some(typeck) = &self.typeck_results else {
            return false;
        };
        let ObligationCauseCode::QuestionMark = obligation.cause.code().peel_derives() else {
            return false;
        };
        let self_ty = trait_pred.skip_binder().self_ty();
        let found_ty = trait_pred.skip_binder().trait_ref.args.get(1).and_then(|a| a.as_type());
        self.note_missing_impl_for_question_mark(err, self_ty, found_ty, trait_pred);

        let mut prev_ty = self.resolve_vars_if_possible(
            typeck.expr_ty_adjusted_opt(expr).unwrap_or(Ty::new_misc_error(self.tcx)),
        );

        // We always look at the `E` type, because that's the only one affected by `?`. If the
        // incorrect `Result<T, E>` is because of the `T`, we'll get an E0308 on the whole
        // expression, after the `?` has "unwrapped" the `T`.
        let get_e_type = |prev_ty: Ty<'tcx>| -> Option<Ty<'tcx>> {
            let ty::Adt(def, args) = prev_ty.kind() else {
                return None;
            };
            let Some(arg) = args.get(1) else {
                return None;
            };
            if !self.tcx.is_diagnostic_item(sym::Result, def.did()) {
                return None;
            }
            arg.as_type()
        };

        let mut suggested = false;
        let mut chain = vec![];

        // The following logic is similar to `point_at_chain`, but that's focused on associated types
        let mut expr = expr;
        while let hir::ExprKind::MethodCall(path_segment, rcvr_expr, args, span) = expr.kind {
            // Point at every method call in the chain with the `Result` type.
            // let foo = bar.iter().map(mapper)?;
            //               ------ -----------
            expr = rcvr_expr;
            chain.push((span, prev_ty));

            let next_ty = self.resolve_vars_if_possible(
                typeck.expr_ty_adjusted_opt(expr).unwrap_or(Ty::new_misc_error(self.tcx)),
            );

            let is_diagnostic_item = |symbol: Symbol, ty: Ty<'tcx>| {
                let ty::Adt(def, _) = ty.kind() else {
                    return false;
                };
                self.tcx.is_diagnostic_item(symbol, def.did())
            };
            // For each method in the chain, see if this is `Result::map_err` or
            // `Option::ok_or_else` and if it is, see if the closure passed to it has an incorrect
            // trailing `;`.
            if let Some(ty) = get_e_type(prev_ty)
                && let Some(found_ty) = found_ty
                // Ideally we would instead use `FnCtxt::lookup_method_for_diagnostic` for 100%
                // accurate check, but we are in the wrong stage to do that and looking for
                // `Result::map_err` by checking the Self type and the path segment is enough.
                // sym::ok_or_else
                && (
                    ( // Result::map_err
                        path_segment.ident.name == sym::map_err
                            && is_diagnostic_item(sym::Result, next_ty)
                    ) || ( // Option::ok_or_else
                        path_segment.ident.name == sym::ok_or_else
                            && is_diagnostic_item(sym::Option, next_ty)
                    )
                )
                // Found `Result<_, ()>?`
                && let ty::Tuple(tys) = found_ty.kind()
                && tys.is_empty()
                // The current method call returns `Result<_, ()>`
                && self.can_eq(obligation.param_env, ty, found_ty)
                // There's a single argument in the method call and it is a closure
                && let [arg] = args
                && let hir::ExprKind::Closure(closure) = arg.kind
                // The closure has a block for its body with no tail expression
                && let body = self.tcx.hir_body(closure.body)
                && let hir::ExprKind::Block(block, _) = body.value.kind
                && let None = block.expr
                // The last statement is of a type that can be converted to the return error type
                && let [.., stmt] = block.stmts
                && let hir::StmtKind::Semi(expr) = stmt.kind
                && let expr_ty = self.resolve_vars_if_possible(
                    typeck.expr_ty_adjusted_opt(expr)
                        .unwrap_or(Ty::new_misc_error(self.tcx)),
                )
                && self
                    .infcx
                    .type_implements_trait(
                        self.tcx.get_diagnostic_item(sym::From).unwrap(),
                        [self_ty, expr_ty],
                        obligation.param_env,
                    )
                    .must_apply_modulo_regions()
            {
                suggested = true;
                err.span_suggestion_short(
                    stmt.span.with_lo(expr.span.hi()),
                    "remove this semicolon",
                    String::new(),
                    Applicability::MachineApplicable,
                );
            }

            prev_ty = next_ty;

            if let hir::ExprKind::Path(hir::QPath::Resolved(None, path)) = expr.kind
                && let hir::Path { res: hir::def::Res::Local(hir_id), .. } = path
                && let hir::Node::Pat(binding) = self.tcx.hir_node(*hir_id)
            {
                let parent = self.tcx.parent_hir_node(binding.hir_id);
                // We've reached the root of the method call chain...
                if let hir::Node::LetStmt(local) = parent
                    && let Some(binding_expr) = local.init
                {
                    // ...and it is a binding. Get the binding creation and continue the chain.
                    expr = binding_expr;
                }
                if let hir::Node::Param(_param) = parent {
                    // ...and it is an fn argument.
                    break;
                }
            }
        }
        // `expr` is now the "root" expression of the method call chain, which can be any
        // expression kind, like a method call or a path. If this expression is `Result<T, E>` as
        // well, then we also point at it.
        prev_ty = self.resolve_vars_if_possible(
            typeck.expr_ty_adjusted_opt(expr).unwrap_or(Ty::new_misc_error(self.tcx)),
        );
        chain.push((expr.span, prev_ty));

        let mut prev = None;
        for (span, err_ty) in chain.into_iter().rev() {
            let err_ty = get_e_type(err_ty);
            let err_ty = match (err_ty, prev) {
                (Some(err_ty), Some(prev)) if !self.can_eq(obligation.param_env, err_ty, prev) => {
                    err_ty
                }
                (Some(err_ty), None) => err_ty,
                _ => {
                    prev = err_ty;
                    continue;
                }
            };
            if self
                .infcx
                .type_implements_trait(
                    self.tcx.get_diagnostic_item(sym::From).unwrap(),
                    [self_ty, err_ty],
                    obligation.param_env,
                )
                .must_apply_modulo_regions()
            {
                if !suggested {
                    err.span_label(span, format!("this has type `Result<_, {err_ty}>`"));
                }
            } else {
                err.span_label(
                    span,
                    format!(
                        "this can't be annotated with `?` because it has type `Result<_, {err_ty}>`",
                    ),
                );
            }
            prev = Some(err_ty);
        }
        suggested
    }

    fn note_missing_impl_for_question_mark(
        &self,
        err: &mut Diag<'_>,
        self_ty: Ty<'_>,
        found_ty: Option<Ty<'_>>,
        trait_pred: ty::PolyTraitPredicate<'tcx>,
    ) {
        match (self_ty.kind(), found_ty) {
            (ty::Adt(def, _), Some(ty))
                if let ty::Adt(found, _) = ty.kind()
                    && def.did().is_local()
                    && found.did().is_local() =>
            {
                err.span_note(
                    self.tcx.def_span(def.did()),
                    format!("`{self_ty}` needs to implement `From<{ty}>`"),
                );
                err.span_note(
                    self.tcx.def_span(found.did()),
                    format!("alternatively, `{ty}` needs to implement `Into<{self_ty}>`"),
                );
            }
            (ty::Adt(def, _), None) if def.did().is_local() => {
                err.span_note(
                    self.tcx.def_span(def.did()),
                    format!(
                        "`{self_ty}` needs to implement `{}`",
                        trait_pred.skip_binder().trait_ref.print_only_trait_path(),
                    ),
                );
            }
            (ty::Adt(def, _), Some(ty)) if def.did().is_local() => {
                err.span_note(
                    self.tcx.def_span(def.did()),
                    format!("`{self_ty}` needs to implement `From<{ty}>`"),
                );
            }
            (_, Some(ty))
                if let ty::Adt(def, _) = ty.kind()
                    && def.did().is_local() =>
            {
                err.span_note(
                    self.tcx.def_span(def.did()),
                    format!("`{ty}` needs to implement `Into<{self_ty}>`"),
                );
            }
            _ => {}
        }
    }

    fn report_const_param_not_wf(
        &self,
        ty: Ty<'tcx>,
        obligation: &PredicateObligation<'tcx>,
    ) -> Diag<'a> {
        let span = obligation.cause.span;

        let mut diag = match ty.kind() {
            ty::Float(_) => {
                struct_span_code_err!(
                    self.dcx(),
                    span,
                    E0741,
                    "`{ty}` is forbidden as the type of a const generic parameter",
                )
            }
            ty::FnPtr(..) => {
                struct_span_code_err!(
                    self.dcx(),
                    span,
                    E0741,
                    "using function pointers as const generic parameters is forbidden",
                )
            }
            ty::RawPtr(_, _) => {
                struct_span_code_err!(
                    self.dcx(),
                    span,
                    E0741,
                    "using raw pointers as const generic parameters is forbidden",
                )
            }
            ty::Adt(def, _) => {
                // We should probably see if we're *allowed* to derive `ConstParamTy` on the type...
                let mut diag = struct_span_code_err!(
                    self.dcx(),
                    span,
                    E0741,
                    "`{ty}` must implement `ConstParamTy` to be used as the type of a const generic parameter",
                );
                // Only suggest derive if this isn't a derived obligation,
                // and the struct is local.
                if let Some(span) = self.tcx.hir_span_if_local(def.did())
                    && obligation.cause.code().parent().is_none()
                {
                    if ty.is_structural_eq_shallow(self.tcx) {
                        diag.span_suggestion(
                            span,
                            "add `#[derive(ConstParamTy)]` to the struct",
                            "#[derive(ConstParamTy)]\n",
                            Applicability::MachineApplicable,
                        );
                    } else {
                        // FIXME(adt_const_params): We should check there's not already an
                        // overlapping `Eq`/`PartialEq` impl.
                        diag.span_suggestion(
                            span,
                            "add `#[derive(ConstParamTy, PartialEq, Eq)]` to the struct",
                            "#[derive(ConstParamTy, PartialEq, Eq)]\n",
                            Applicability::MachineApplicable,
                        );
                    }
                }
                diag
            }
            _ => {
                struct_span_code_err!(
                    self.dcx(),
                    span,
                    E0741,
                    "`{ty}` can't be used as a const parameter type",
                )
            }
        };

        let mut code = obligation.cause.code();
        let mut pred = obligation.predicate.as_trait_clause();
        while let Some((next_code, next_pred)) = code.parent_with_predicate() {
            if let Some(pred) = pred {
                self.enter_forall(pred, |pred| {
                    diag.note(format!(
                        "`{}` must implement `{}`, but it does not",
                        pred.self_ty(),
                        pred.print_modifiers_and_trait_path()
                    ));
                })
            }
            code = next_code;
            pred = next_pred;
        }

        diag
    }
}

impl<'a, 'tcx> TypeErrCtxt<'a, 'tcx> {
    fn can_match_trait(
        &self,
        param_env: ty::ParamEnv<'tcx>,
        goal: ty::TraitPredicate<'tcx>,
        assumption: ty::PolyTraitPredicate<'tcx>,
    ) -> bool {
        // Fast path
        if goal.polarity != assumption.polarity() {
            return false;
        }

        let trait_assumption = self.instantiate_binder_with_fresh_vars(
            DUMMY_SP,
            infer::BoundRegionConversionTime::HigherRankedType,
            assumption,
        );

        self.can_eq(param_env, goal.trait_ref, trait_assumption.trait_ref)
    }

    fn can_match_projection(
        &self,
        param_env: ty::ParamEnv<'tcx>,
        goal: ty::ProjectionPredicate<'tcx>,
        assumption: ty::PolyProjectionPredicate<'tcx>,
    ) -> bool {
        let assumption = self.instantiate_binder_with_fresh_vars(
            DUMMY_SP,
            infer::BoundRegionConversionTime::HigherRankedType,
            assumption,
        );

        self.can_eq(param_env, goal.projection_term, assumption.projection_term)
            && self.can_eq(param_env, goal.term, assumption.term)
    }

    // returns if `cond` not occurring implies that `error` does not occur - i.e., that
    // `error` occurring implies that `cond` occurs.
    #[instrument(level = "debug", skip(self), ret)]
    pub(super) fn error_implies(
        &self,
        cond: Goal<'tcx, ty::Predicate<'tcx>>,
        error: Goal<'tcx, ty::Predicate<'tcx>>,
    ) -> bool {
        if cond == error {
            return true;
        }

        // FIXME: We could be smarter about this, i.e. if cond's param-env is a
        // subset of error's param-env. This only matters when binders will carry
        // predicates though, and obviously only matters for error reporting.
        if cond.param_env != error.param_env {
            return false;
        }
        let param_env = error.param_env;

        if let Some(error) = error.predicate.as_trait_clause() {
            self.enter_forall(error, |error| {
                elaborate(self.tcx, std::iter::once(cond.predicate))
                    .filter_map(|implied| implied.as_trait_clause())
                    .any(|implied| self.can_match_trait(param_env, error, implied))
            })
        } else if let Some(error) = error.predicate.as_projection_clause() {
            self.enter_forall(error, |error| {
                elaborate(self.tcx, std::iter::once(cond.predicate))
                    .filter_map(|implied| implied.as_projection_clause())
                    .any(|implied| self.can_match_projection(param_env, error, implied))
            })
        } else {
            false
        }
    }

    #[instrument(level = "debug", skip_all)]
    pub(super) fn report_projection_error(
        &self,
        obligation: &PredicateObligation<'tcx>,
        error: &MismatchedProjectionTypes<'tcx>,
    ) -> ErrorGuaranteed {
        let predicate = self.resolve_vars_if_possible(obligation.predicate);

        if let Err(e) = predicate.error_reported() {
            return e;
        }

        self.probe(|_| {
            // try to find the mismatched types to report the error with.
            //
            // this can fail if the problem was higher-ranked, in which
            // cause I have no idea for a good error message.
            let bound_predicate = predicate.kind();
            let (values, err) = match bound_predicate.skip_binder() {
                ty::PredicateKind::Clause(ty::ClauseKind::Projection(data)) => {
                    let ocx = ObligationCtxt::new(self);

                    let data = self.instantiate_binder_with_fresh_vars(
                        obligation.cause.span,
                        infer::BoundRegionConversionTime::HigherRankedType,
                        bound_predicate.rebind(data),
                    );
                    let unnormalized_term = data.projection_term.to_term(self.tcx);
                    // FIXME(-Znext-solver): For diagnostic purposes, it would be nice
                    // to deeply normalize this type.
                    let normalized_term =
                        ocx.normalize(&obligation.cause, obligation.param_env, unnormalized_term);

                    // constrain inference variables a bit more to nested obligations from normalize so
                    // we can have more helpful errors.
                    //
                    // we intentionally drop errors from normalization here,
                    // since the normalization is just done to improve the error message.
                    let _ = ocx.select_where_possible();

                    if let Err(new_err) =
                        ocx.eq(&obligation.cause, obligation.param_env, data.term, normalized_term)
                    {
                        (
                            Some((
                                data.projection_term,
                                self.resolve_vars_if_possible(normalized_term),
                                data.term,
                            )),
                            new_err,
                        )
                    } else {
                        (None, error.err)
                    }
                }
                ty::PredicateKind::AliasRelate(lhs, rhs, _) => {
                    let derive_better_type_error =
                        |alias_term: ty::AliasTerm<'tcx>, expected_term: ty::Term<'tcx>| {
                            let ocx = ObligationCtxt::new(self);

                            let Ok(normalized_term) = ocx.structurally_normalize_term(
                                &ObligationCause::dummy(),
                                obligation.param_env,
                                alias_term.to_term(self.tcx),
                            ) else {
                                return None;
                            };

                            if let Err(terr) = ocx.eq(
                                &ObligationCause::dummy(),
                                obligation.param_env,
                                expected_term,
                                normalized_term,
                            ) {
                                Some((terr, self.resolve_vars_if_possible(normalized_term)))
                            } else {
                                None
                            }
                        };

                    if let Some(lhs) = lhs.to_alias_term()
                        && let Some((better_type_err, expected_term)) =
                            derive_better_type_error(lhs, rhs)
                    {
                        (
                            Some((lhs, self.resolve_vars_if_possible(expected_term), rhs)),
                            better_type_err,
                        )
                    } else if let Some(rhs) = rhs.to_alias_term()
                        && let Some((better_type_err, expected_term)) =
                            derive_better_type_error(rhs, lhs)
                    {
                        (
                            Some((rhs, self.resolve_vars_if_possible(expected_term), lhs)),
                            better_type_err,
                        )
                    } else {
                        (None, error.err)
                    }
                }
                _ => (None, error.err),
            };

            let mut file = None;
            let (msg, span, closure_span) = values
                .and_then(|(predicate, normalized_term, expected_term)| {
                    self.maybe_detailed_projection_msg(
                        obligation.cause.span,
                        predicate,
                        normalized_term,
                        expected_term,
                        &mut file,
                    )
                })
                .unwrap_or_else(|| {
                    (
                        with_forced_trimmed_paths!(format!(
                            "type mismatch resolving `{}`",
                            self.tcx
                                .short_string(self.resolve_vars_if_possible(predicate), &mut file),
                        )),
                        obligation.cause.span,
                        None,
                    )
                });
            let mut diag = struct_span_code_err!(self.dcx(), span, E0271, "{msg}");
            *diag.long_ty_path() = file;
            if let Some(span) = closure_span {
                // Mark the closure decl so that it is seen even if we are pointing at the return
                // type or expression.
                //
                // error[E0271]: expected `{closure@foo.rs:41:16}` to be a closure that returns
                //               `Unit3`, but it returns `Unit4`
                //   --> $DIR/foo.rs:43:17
                //    |
                // LL |     let v = Unit2.m(
                //    |                   - required by a bound introduced by this call
                // ...
                // LL |             f: |x| {
                //    |                --- /* this span */
                // LL |                 drop(x);
                // LL |                 Unit4
                //    |                 ^^^^^ expected `Unit3`, found `Unit4`
                //    |
                diag.span_label(span, "this closure");
                if !span.overlaps(obligation.cause.span) {
                    // Point at the binding corresponding to the closure where it is used.
                    diag.span_label(obligation.cause.span, "closure used here");
                }
            }

            let secondary_span = self.probe(|_| {
                let ty::PredicateKind::Clause(ty::ClauseKind::Projection(proj)) =
                    predicate.kind().skip_binder()
                else {
                    return None;
                };

                let trait_ref = self.enter_forall_and_leak_universe(
                    predicate.kind().rebind(proj.projection_term.trait_ref(self.tcx)),
                );
                let Ok(Some(ImplSource::UserDefined(impl_data))) =
                    SelectionContext::new(self).select(&obligation.with(self.tcx, trait_ref))
                else {
                    return None;
                };

                let Ok(node) =
                    specialization_graph::assoc_def(self.tcx, impl_data.impl_def_id, proj.def_id())
                else {
                    return None;
                };

                if !node.is_final() {
                    return None;
                }

                match self.tcx.hir_get_if_local(node.item.def_id) {
                    Some(
                        hir::Node::TraitItem(hir::TraitItem {
                            kind: hir::TraitItemKind::Type(_, Some(ty)),
                            ..
                        })
                        | hir::Node::ImplItem(hir::ImplItem {
                            kind: hir::ImplItemKind::Type(ty),
                            ..
                        }),
                    ) => Some((
                        ty.span,
                        with_forced_trimmed_paths!(Cow::from(format!(
                            "type mismatch resolving `{}`",
                            self.tcx.short_string(
                                self.resolve_vars_if_possible(predicate),
                                diag.long_ty_path()
                            ),
                        ))),
                        true,
                    )),
                    _ => None,
                }
            });

            self.note_type_err(
                &mut diag,
                &obligation.cause,
                secondary_span,
                values.map(|(_, normalized_ty, expected_ty)| {
                    obligation.param_env.and(infer::ValuePairs::Terms(ExpectedFound::new(
                        expected_ty,
                        normalized_ty,
                    )))
                }),
                err,
                false,
                Some(span),
            );
            self.note_obligation_cause(&mut diag, obligation);
            diag.emit()
        })
    }

    fn maybe_detailed_projection_msg(
        &self,
        mut span: Span,
        projection_term: ty::AliasTerm<'tcx>,
        normalized_ty: ty::Term<'tcx>,
        expected_ty: ty::Term<'tcx>,
        file: &mut Option<PathBuf>,
    ) -> Option<(String, Span, Option<Span>)> {
        let trait_def_id = projection_term.trait_def_id(self.tcx);
        let self_ty = projection_term.self_ty();

        with_forced_trimmed_paths! {
            if self.tcx.is_lang_item(projection_term.def_id, LangItem::FnOnceOutput) {
                let (span, closure_span) = if let ty::Closure(def_id, _) = self_ty.kind() {
                    let def_span = self.tcx.def_span(def_id);
                    if let Some(local_def_id) = def_id.as_local()
                        && let node = self.tcx.hir_node_by_def_id(local_def_id)
                        && let Some(fn_decl) = node.fn_decl()
                        && let Some(id) = node.body_id()
                    {
                        span = match fn_decl.output {
                            hir::FnRetTy::Return(ty) => ty.span,
                            hir::FnRetTy::DefaultReturn(_) => {
                                let body = self.tcx.hir_body(id);
                                match body.value.kind {
                                    hir::ExprKind::Block(
                                        hir::Block { expr: Some(expr), .. },
                                        _,
                                    ) => expr.span,
                                    hir::ExprKind::Block(
                                        hir::Block {
                                            expr: None, stmts: [.., last], ..
                                        },
                                        _,
                                    ) => last.span,
                                    _ => body.value.span,
                                }
                            }
                        };
                    }
                    (span, Some(def_span))
                } else {
                    (span, None)
                };
                let item = match self_ty.kind() {
                    ty::FnDef(def, _) => self.tcx.item_name(*def).to_string(),
                    _ => self.tcx.short_string(self_ty, file),
                };
                Some((format!(
                    "expected `{item}` to return `{expected_ty}`, but it returns `{normalized_ty}`",
                ), span, closure_span))
            } else if self.tcx.is_lang_item(trait_def_id, LangItem::Future) {
                Some((format!(
                    "expected `{self_ty}` to be a future that resolves to `{expected_ty}`, but it \
                     resolves to `{normalized_ty}`"
                ), span, None))
            } else if Some(trait_def_id) == self.tcx.get_diagnostic_item(sym::Iterator) {
                Some((format!(
                    "expected `{self_ty}` to be an iterator that yields `{expected_ty}`, but it \
                     yields `{normalized_ty}`"
                ), span, None))
            } else {
                None
            }
        }
    }

    pub fn fuzzy_match_tys(
        &self,
        mut a: Ty<'tcx>,
        mut b: Ty<'tcx>,
        ignoring_lifetimes: bool,
    ) -> Option<CandidateSimilarity> {
        /// returns the fuzzy category of a given type, or None
        /// if the type can be equated to any type.
        fn type_category(tcx: TyCtxt<'_>, t: Ty<'_>) -> Option<u32> {
            match t.kind() {
                ty::Bool => Some(0),
                ty::Char => Some(1),
                ty::Str => Some(2),
                ty::Adt(def, _) if tcx.is_lang_item(def.did(), LangItem::String) => Some(2),
                ty::Int(..)
                | ty::Uint(..)
                | ty::Float(..)
                | ty::Infer(ty::IntVar(..) | ty::FloatVar(..)) => Some(4),
                ty::Ref(..) | ty::RawPtr(..) => Some(5),
                ty::Array(..) | ty::Slice(..) => Some(6),
                ty::FnDef(..) | ty::FnPtr(..) => Some(7),
                ty::Dynamic(..) => Some(8),
                ty::Closure(..) => Some(9),
                ty::Tuple(..) => Some(10),
                ty::Param(..) => Some(11),
                ty::Alias(ty::Projection, ..) => Some(12),
                ty::Alias(ty::Inherent, ..) => Some(13),
                ty::Alias(ty::Opaque, ..) => Some(14),
                ty::Alias(ty::Free, ..) => Some(15),
                ty::Never => Some(16),
                ty::Adt(..) => Some(17),
                ty::Coroutine(..) => Some(18),
                ty::Foreign(..) => Some(19),
                ty::CoroutineWitness(..) => Some(20),
                ty::CoroutineClosure(..) => Some(21),
                ty::Pat(..) => Some(22),
                ty::UnsafeBinder(..) => Some(23),
                ty::Placeholder(..) | ty::Bound(..) | ty::Infer(..) | ty::Error(_) => None,
            }
        }

        let strip_references = |mut t: Ty<'tcx>| -> Ty<'tcx> {
            loop {
                match t.kind() {
                    ty::Ref(_, inner, _) | ty::RawPtr(inner, _) => t = *inner,
                    _ => break t,
                }
            }
        };

        if !ignoring_lifetimes {
            a = strip_references(a);
            b = strip_references(b);
        }

        let cat_a = type_category(self.tcx, a)?;
        let cat_b = type_category(self.tcx, b)?;
        if a == b {
            Some(CandidateSimilarity::Exact { ignoring_lifetimes })
        } else if cat_a == cat_b {
            match (a.kind(), b.kind()) {
                (ty::Adt(def_a, _), ty::Adt(def_b, _)) => def_a == def_b,
                (ty::Foreign(def_a), ty::Foreign(def_b)) => def_a == def_b,
                // Matching on references results in a lot of unhelpful
                // suggestions, so let's just not do that for now.
                //
                // We still upgrade successful matches to `ignoring_lifetimes: true`
                // to prioritize that impl.
                (ty::Ref(..) | ty::RawPtr(..), ty::Ref(..) | ty::RawPtr(..)) => {
                    self.fuzzy_match_tys(a, b, true).is_some()
                }
                _ => true,
            }
            .then_some(CandidateSimilarity::Fuzzy { ignoring_lifetimes })
        } else if ignoring_lifetimes {
            None
        } else {
            self.fuzzy_match_tys(a, b, true)
        }
    }

    pub(super) fn describe_closure(&self, kind: hir::ClosureKind) -> &'static str {
        match kind {
            hir::ClosureKind::Closure => "a closure",
            hir::ClosureKind::Coroutine(hir::CoroutineKind::Coroutine(_)) => "a coroutine",
            hir::ClosureKind::Coroutine(hir::CoroutineKind::Desugared(
                hir::CoroutineDesugaring::Async,
                hir::CoroutineSource::Block,
            )) => "an async block",
            hir::ClosureKind::Coroutine(hir::CoroutineKind::Desugared(
                hir::CoroutineDesugaring::Async,
                hir::CoroutineSource::Fn,
            )) => "an async function",
            hir::ClosureKind::Coroutine(hir::CoroutineKind::Desugared(
                hir::CoroutineDesugaring::Async,
                hir::CoroutineSource::Closure,
            ))
            | hir::ClosureKind::CoroutineClosure(hir::CoroutineDesugaring::Async) => {
                "an async closure"
            }
            hir::ClosureKind::Coroutine(hir::CoroutineKind::Desugared(
                hir::CoroutineDesugaring::AsyncGen,
                hir::CoroutineSource::Block,
            )) => "an async gen block",
            hir::ClosureKind::Coroutine(hir::CoroutineKind::Desugared(
                hir::CoroutineDesugaring::AsyncGen,
                hir::CoroutineSource::Fn,
            )) => "an async gen function",
            hir::ClosureKind::Coroutine(hir::CoroutineKind::Desugared(
                hir::CoroutineDesugaring::AsyncGen,
                hir::CoroutineSource::Closure,
            ))
            | hir::ClosureKind::CoroutineClosure(hir::CoroutineDesugaring::AsyncGen) => {
                "an async gen closure"
            }
            hir::ClosureKind::Coroutine(hir::CoroutineKind::Desugared(
                hir::CoroutineDesugaring::Gen,
                hir::CoroutineSource::Block,
            )) => "a gen block",
            hir::ClosureKind::Coroutine(hir::CoroutineKind::Desugared(
                hir::CoroutineDesugaring::Gen,
                hir::CoroutineSource::Fn,
            )) => "a gen function",
            hir::ClosureKind::Coroutine(hir::CoroutineKind::Desugared(
                hir::CoroutineDesugaring::Gen,
                hir::CoroutineSource::Closure,
            ))
            | hir::ClosureKind::CoroutineClosure(hir::CoroutineDesugaring::Gen) => "a gen closure",
        }
    }

    pub(super) fn find_similar_impl_candidates(
        &self,
        trait_pred: ty::PolyTraitPredicate<'tcx>,
    ) -> Vec<ImplCandidate<'tcx>> {
        let mut candidates: Vec<_> = self
            .tcx
            .all_impls(trait_pred.def_id())
            .filter_map(|def_id| {
                let imp = self.tcx.impl_trait_header(def_id).unwrap();
                if imp.polarity != ty::ImplPolarity::Positive
                    || !self.tcx.is_user_visible_dep(def_id.krate)
                {
                    return None;
                }
                let imp = imp.trait_ref.skip_binder();

                self.fuzzy_match_tys(trait_pred.skip_binder().self_ty(), imp.self_ty(), false).map(
                    |similarity| ImplCandidate { trait_ref: imp, similarity, impl_def_id: def_id },
                )
            })
            .collect();
        if candidates.iter().any(|c| matches!(c.similarity, CandidateSimilarity::Exact { .. })) {
            // If any of the candidates is a perfect match, we don't want to show all of them.
            // This is particularly relevant for the case of numeric types (as they all have the
            // same category).
            candidates.retain(|c| matches!(c.similarity, CandidateSimilarity::Exact { .. }));
        }
        candidates
    }

    pub(super) fn report_similar_impl_candidates(
        &self,
        impl_candidates: &[ImplCandidate<'tcx>],
        trait_pred: ty::PolyTraitPredicate<'tcx>,
        body_def_id: LocalDefId,
        err: &mut Diag<'_>,
        other: bool,
        param_env: ty::ParamEnv<'tcx>,
    ) -> bool {
        let alternative_candidates = |def_id: DefId| {
            let mut impl_candidates: Vec<_> = self
                .tcx
                .all_impls(def_id)
                // ignore `do_not_recommend` items
                .filter(|def_id| !self.tcx.do_not_recommend_impl(*def_id))
                // Ignore automatically derived impls and `!Trait` impls.
                .filter_map(|def_id| self.tcx.impl_trait_header(def_id))
                .filter_map(|header| {
                    (header.polarity != ty::ImplPolarity::Negative
                        || self.tcx.is_automatically_derived(def_id))
                    .then(|| header.trait_ref.instantiate_identity())
                })
                .filter(|trait_ref| {
                    let self_ty = trait_ref.self_ty();
                    // Avoid mentioning type parameters.
                    if let ty::Param(_) = self_ty.kind() {
                        false
                    }
                    // Avoid mentioning types that are private to another crate
                    else if let ty::Adt(def, _) = self_ty.peel_refs().kind() {
                        // FIXME(compiler-errors): This could be generalized, both to
                        // be more granular, and probably look past other `#[fundamental]`
                        // types, too.
                        self.tcx.visibility(def.did()).is_accessible_from(body_def_id, self.tcx)
                    } else {
                        true
                    }
                })
                .collect();

            impl_candidates.sort_by_key(|tr| tr.to_string());
            impl_candidates.dedup();
            impl_candidates
        };

        // We'll check for the case where the reason for the mismatch is that the trait comes from
        // one crate version and the type comes from another crate version, even though they both
        // are from the same crate.
        let trait_def_id = trait_pred.def_id();
        let trait_name = self.tcx.item_name(trait_def_id);
        let crate_name = self.tcx.crate_name(trait_def_id.krate);
        if let Some(other_trait_def_id) = self.tcx.all_traits().find(|def_id| {
            trait_name == self.tcx.item_name(trait_def_id)
                && trait_def_id.krate != def_id.krate
                && crate_name == self.tcx.crate_name(def_id.krate)
        }) {
            // We've found two different traits with the same name, same crate name, but
            // different crate `DefId`. We highlight the traits.

            let found_type =
                if let ty::Adt(def, _) = trait_pred.self_ty().skip_binder().peel_refs().kind() {
                    Some(def.did())
                } else {
                    None
                };
            let candidates = if impl_candidates.is_empty() {
                alternative_candidates(trait_def_id)
            } else {
                impl_candidates.into_iter().map(|cand| cand.trait_ref).collect()
            };
            let mut span: MultiSpan = self.tcx.def_span(trait_def_id).into();
            span.push_span_label(self.tcx.def_span(trait_def_id), "this is the required trait");
            for (sp, label) in [trait_def_id, other_trait_def_id]
                .iter()
                // The current crate-version might depend on another version of the same crate
                // (Think "semver-trick"). Do not call `extern_crate` in that case for the local
                // crate as that doesn't make sense and ICEs (#133563).
                .filter(|def_id| !def_id.is_local())
                .filter_map(|def_id| self.tcx.extern_crate(def_id.krate))
                .map(|data| {
                    let dependency = if data.dependency_of == LOCAL_CRATE {
                        "direct dependency of the current crate".to_string()
                    } else {
                        let dep = self.tcx.crate_name(data.dependency_of);
                        format!("dependency of crate `{dep}`")
                    };
                    (
                        data.span,
                        format!("one version of crate `{crate_name}` used here, as a {dependency}"),
                    )
                })
            {
                span.push_span_label(sp, label);
            }
            let mut points_at_type = false;
            if let Some(found_type) = found_type {
                span.push_span_label(
                    self.tcx.def_span(found_type),
                    "this type doesn't implement the required trait",
                );
                for trait_ref in candidates {
                    if let ty::Adt(def, _) = trait_ref.self_ty().peel_refs().kind()
                        && let candidate_def_id = def.did()
                        && let Some(name) = self.tcx.opt_item_name(candidate_def_id)
                        && let Some(found) = self.tcx.opt_item_name(found_type)
                        && name == found
                        && candidate_def_id.krate != found_type.krate
                        && self.tcx.crate_name(candidate_def_id.krate)
                            == self.tcx.crate_name(found_type.krate)
                    {
                        // A candidate was found of an item with the same name, from two separate
                        // versions of the same crate, let's clarify.
                        let candidate_span = self.tcx.def_span(candidate_def_id);
                        span.push_span_label(
                            candidate_span,
                            "this type implements the required trait",
                        );
                        points_at_type = true;
                    }
                }
            }
            span.push_span_label(self.tcx.def_span(other_trait_def_id), "this is the found trait");
            err.highlighted_span_note(
                span,
                vec![
                    StringPart::normal("there are ".to_string()),
                    StringPart::highlighted("multiple different versions".to_string()),
                    StringPart::normal(" of crate `".to_string()),
                    StringPart::highlighted(format!("{crate_name}")),
                    StringPart::normal("` in the dependency graph\n".to_string()),
                ],
            );
            if points_at_type {
                // We only clarify that the same type from different crate versions are not the
                // same when we *find* the same type coming from different crate versions, otherwise
                // it could be that it was a type provided by a different crate than the one that
                // provides the trait, and mentioning this adds verbosity without clarification.
                err.highlighted_note(vec![
                    StringPart::normal(
                        "two types coming from two different versions of the same crate are \
                         different types "
                            .to_string(),
                    ),
                    StringPart::highlighted("even if they look the same".to_string()),
                ]);
            }
            err.highlighted_help(vec![
                StringPart::normal("you can use `".to_string()),
                StringPart::highlighted("cargo tree".to_string()),
                StringPart::normal("` to explore your dependency tree".to_string()),
            ]);
            return true;
        }

        if let [single] = &impl_candidates {
            // If we have a single implementation, try to unify it with the trait ref
            // that failed. This should uncover a better hint for what *is* implemented.
            if self.probe(|_| {
                let ocx = ObligationCtxt::new(self);

                self.enter_forall(trait_pred, |obligation_trait_ref| {
                    let impl_args = self.fresh_args_for_item(DUMMY_SP, single.impl_def_id);
                    let impl_trait_ref = ocx.normalize(
                        &ObligationCause::dummy(),
                        param_env,
                        ty::EarlyBinder::bind(single.trait_ref).instantiate(self.tcx, impl_args),
                    );

                    ocx.register_obligations(
                        self.tcx
                            .predicates_of(single.impl_def_id)
                            .instantiate(self.tcx, impl_args)
                            .into_iter()
                            .map(|(clause, _)| {
                                Obligation::new(
                                    self.tcx,
                                    ObligationCause::dummy(),
                                    param_env,
                                    clause,
                                )
                            }),
                    );
                    if !ocx.select_where_possible().is_empty() {
                        return false;
                    }

                    let mut terrs = vec![];
                    for (obligation_arg, impl_arg) in
                        std::iter::zip(obligation_trait_ref.trait_ref.args, impl_trait_ref.args)
                    {
                        if (obligation_arg, impl_arg).references_error() {
                            return false;
                        }
                        if let Err(terr) =
                            ocx.eq(&ObligationCause::dummy(), param_env, impl_arg, obligation_arg)
                        {
                            terrs.push(terr);
                        }
                        if !ocx.select_where_possible().is_empty() {
                            return false;
                        }
                    }

                    // Literally nothing unified, just give up.
                    if terrs.len() == impl_trait_ref.args.len() {
                        return false;
                    }

                    let impl_trait_ref = self.resolve_vars_if_possible(impl_trait_ref);
                    if impl_trait_ref.references_error() {
                        return false;
                    }

                    if let [child, ..] = &err.children[..]
                        && child.level == Level::Help
                        && let Some(line) = child.messages.get(0)
                        && let Some(line) = line.0.as_str()
                        && line.starts_with("the trait")
                        && line.contains("is not implemented for")
                    {
                        // HACK(estebank): we remove the pre-existing
                        // "the trait `X` is not implemented for" note, which only happens if there
                        // was a custom label. We do this because we want that note to always be the
                        // first, and making this logic run earlier will get tricky. For now, we
                        // instead keep the logic the same and modify the already constructed error
                        // to avoid the wording duplication.
                        err.children.remove(0);
                    }

                    let traits = self.cmp_traits(
                        obligation_trait_ref.def_id(),
                        &obligation_trait_ref.trait_ref.args[1..],
                        impl_trait_ref.def_id,
                        &impl_trait_ref.args[1..],
                    );
                    let traits_content = (traits.0.content(), traits.1.content());
                    let types = self.cmp(obligation_trait_ref.self_ty(), impl_trait_ref.self_ty());
                    let types_content = (types.0.content(), types.1.content());
                    let mut msg = vec![StringPart::normal("the trait `")];
                    if traits_content.0 == traits_content.1 {
                        msg.push(StringPart::normal(
                            impl_trait_ref.print_trait_sugared().to_string(),
                        ));
                    } else {
                        msg.extend(traits.0.0);
                    }
                    msg.extend([
                        StringPart::normal("` "),
                        StringPart::highlighted("is not"),
                        StringPart::normal(" implemented for `"),
                    ]);
                    if types_content.0 == types_content.1 {
                        let ty = self
                            .tcx
                            .short_string(obligation_trait_ref.self_ty(), err.long_ty_path());
                        msg.push(StringPart::normal(ty));
                    } else {
                        msg.extend(types.0.0);
                    }
                    msg.push(StringPart::normal("`"));
                    if types_content.0 == types_content.1 {
                        msg.push(StringPart::normal("\nbut trait `"));
                        msg.extend(traits.1.0);
                        msg.extend([
                            StringPart::normal("` "),
                            StringPart::highlighted("is"),
                            StringPart::normal(" implemented for it"),
                        ]);
                    } else if traits_content.0 == traits_content.1 {
                        msg.extend([
                            StringPart::normal("\nbut it "),
                            StringPart::highlighted("is"),
                            StringPart::normal(" implemented for `"),
                        ]);
                        msg.extend(types.1.0);
                        msg.push(StringPart::normal("`"));
                    } else {
                        msg.push(StringPart::normal("\nbut trait `"));
                        msg.extend(traits.1.0);
                        msg.extend([
                            StringPart::normal("` "),
                            StringPart::highlighted("is"),
                            StringPart::normal(" implemented for `"),
                        ]);
                        msg.extend(types.1.0);
                        msg.push(StringPart::normal("`"));
                    }
                    err.highlighted_help(msg);

                    if let [TypeError::Sorts(exp_found)] = &terrs[..] {
                        let exp_found = self.resolve_vars_if_possible(*exp_found);
                        err.highlighted_help(vec![
                            StringPart::normal("for that trait implementation, "),
                            StringPart::normal("expected `"),
                            StringPart::highlighted(exp_found.expected.to_string()),
                            StringPart::normal("`, found `"),
                            StringPart::highlighted(exp_found.found.to_string()),
                            StringPart::normal("`"),
                        ]);
                        self.suggest_function_pointers_impl(None, &exp_found, err);
                    }

                    true
                })
            }) {
                return true;
            }
        }

        let other = if other { "other " } else { "" };
        let report = |mut candidates: Vec<TraitRef<'tcx>>, err: &mut Diag<'_>| {
            candidates.retain(|tr| !tr.references_error());
            if candidates.is_empty() {
                return false;
            }
            if let &[cand] = &candidates[..] {
                if self.tcx.is_diagnostic_item(sym::FromResidual, cand.def_id)
                    && !self.tcx.features().enabled(sym::try_trait_v2)
                {
                    return false;
                }
                let (desc, mention_castable) =
                    match (cand.self_ty().kind(), trait_pred.self_ty().skip_binder().kind()) {
                        (ty::FnPtr(..), ty::FnDef(..)) => {
                            (" implemented for fn pointer `", ", cast using `as`")
                        }
                        (ty::FnPtr(..), _) => (" implemented for fn pointer `", ""),
                        _ => (" implemented for `", ""),
                    };
                err.highlighted_help(vec![
                    StringPart::normal(format!("the trait `{}` ", cand.print_trait_sugared())),
                    StringPart::highlighted("is"),
                    StringPart::normal(desc),
                    StringPart::highlighted(cand.self_ty().to_string()),
                    StringPart::normal("`"),
                    StringPart::normal(mention_castable),
                ]);
                return true;
            }
            let trait_ref = TraitRef::identity(self.tcx, candidates[0].def_id);
            // Check if the trait is the same in all cases. If so, we'll only show the type.
            let mut traits: Vec<_> =
                candidates.iter().map(|c| c.print_only_trait_path().to_string()).collect();
            traits.sort();
            traits.dedup();
            // FIXME: this could use a better heuristic, like just checking
            // that args[1..] is the same.
            let all_traits_equal = traits.len() == 1;

            let candidates: Vec<String> = candidates
                .into_iter()
                .map(|c| {
                    if all_traits_equal {
                        format!("\n  {}", c.self_ty())
                    } else {
                        format!("\n  `{}` implements `{}`", c.self_ty(), c.print_only_trait_path())
                    }
                })
                .collect();

            let end = if candidates.len() <= 9 || self.tcx.sess.opts.verbose {
                candidates.len()
            } else {
                8
            };
            err.help(format!(
                "the following {other}types implement trait `{}`:{}{}",
                trait_ref.print_trait_sugared(),
                candidates[..end].join(""),
                if candidates.len() > 9 && !self.tcx.sess.opts.verbose {
                    format!("\nand {} others", candidates.len() - 8)
                } else {
                    String::new()
                }
            ));
            true
        };

        // we filter before checking if `impl_candidates` is empty
        // to get the fallback solution if we filtered out any impls
        let impl_candidates = impl_candidates
            .into_iter()
            .cloned()
            .filter(|cand| !self.tcx.do_not_recommend_impl(cand.impl_def_id))
            .collect::<Vec<_>>();

        let def_id = trait_pred.def_id();
        if impl_candidates.is_empty() {
            if self.tcx.trait_is_auto(def_id)
                || self.tcx.lang_items().iter().any(|(_, id)| id == def_id)
                || self.tcx.get_diagnostic_name(def_id).is_some()
            {
                // Mentioning implementers of `Copy`, `Debug` and friends is not useful.
                return false;
            }
            return report(alternative_candidates(def_id), err);
        }

        // Sort impl candidates so that ordering is consistent for UI tests.
        // because the ordering of `impl_candidates` may not be deterministic:
        // https://github.com/rust-lang/rust/pull/57475#issuecomment-455519507
        //
        // Prefer more similar candidates first, then sort lexicographically
        // by their normalized string representation.
        let mut impl_candidates: Vec<_> = impl_candidates
            .iter()
            .cloned()
            .filter(|cand| !cand.trait_ref.references_error())
            .map(|mut cand| {
                // Normalize the trait ref in its *own* param-env so
                // that consts are folded and any trivial projections
                // are normalized.
                cand.trait_ref = self
                    .tcx
                    .try_normalize_erasing_regions(
                        ty::TypingEnv::non_body_analysis(self.tcx, cand.impl_def_id),
                        cand.trait_ref,
                    )
                    .unwrap_or(cand.trait_ref);
                cand
            })
            .collect();
        impl_candidates.sort_by_key(|cand| (cand.similarity, cand.trait_ref.to_string()));
        let mut impl_candidates: Vec<_> =
            impl_candidates.into_iter().map(|cand| cand.trait_ref).collect();
        impl_candidates.dedup();

        report(impl_candidates, err)
    }

    fn report_similar_impl_candidates_for_root_obligation(
        &self,
        obligation: &PredicateObligation<'tcx>,
        trait_predicate: ty::Binder<'tcx, ty::TraitPredicate<'tcx>>,
        body_def_id: LocalDefId,
        err: &mut Diag<'_>,
    ) {
        // This is *almost* equivalent to
        // `obligation.cause.code().peel_derives()`, but it gives us the
        // trait predicate for that corresponding root obligation. This
        // lets us get a derived obligation from a type parameter, like
        // when calling `string.strip_suffix(p)` where `p` is *not* an
        // implementer of `Pattern<'_>`.
        let mut code = obligation.cause.code();
        let mut trait_pred = trait_predicate;
        let mut peeled = false;
        while let Some((parent_code, parent_trait_pred)) = code.parent_with_predicate() {
            code = parent_code;
            if let Some(parent_trait_pred) = parent_trait_pred {
                trait_pred = parent_trait_pred;
                peeled = true;
            }
        }
        let def_id = trait_pred.def_id();
        // Mention *all* the `impl`s for the *top most* obligation, the
        // user might have meant to use one of them, if any found. We skip
        // auto-traits or fundamental traits that might not be exactly what
        // the user might expect to be presented with. Instead this is
        // useful for less general traits.
        if peeled && !self.tcx.trait_is_auto(def_id) && self.tcx.as_lang_item(def_id).is_none() {
            let impl_candidates = self.find_similar_impl_candidates(trait_pred);
            self.report_similar_impl_candidates(
                &impl_candidates,
                trait_pred,
                body_def_id,
                err,
                true,
                obligation.param_env,
            );
        }
    }

    /// Gets the parent trait chain start
    fn get_parent_trait_ref(
        &self,
        code: &ObligationCauseCode<'tcx>,
    ) -> Option<(Ty<'tcx>, Option<Span>)> {
        match code {
            ObligationCauseCode::BuiltinDerived(data) => {
                let parent_trait_ref = self.resolve_vars_if_possible(data.parent_trait_pred);
                match self.get_parent_trait_ref(&data.parent_code) {
                    Some(t) => Some(t),
                    None => {
                        let ty = parent_trait_ref.skip_binder().self_ty();
                        let span = TyCategory::from_ty(self.tcx, ty)
                            .map(|(_, def_id)| self.tcx.def_span(def_id));
                        Some((ty, span))
                    }
                }
            }
            ObligationCauseCode::FunctionArg { parent_code, .. } => {
                self.get_parent_trait_ref(parent_code)
            }
            _ => None,
        }
    }

    /// If the `Self` type of the unsatisfied trait `trait_ref` implements a trait
    /// with the same path as `trait_ref`, a help message about
    /// a probable version mismatch is added to `err`
    fn note_version_mismatch(
        &self,
        err: &mut Diag<'_>,
        trait_pred: ty::PolyTraitPredicate<'tcx>,
    ) -> bool {
        let get_trait_impls = |trait_def_id| {
            let mut trait_impls = vec![];
            self.tcx.for_each_relevant_impl(
                trait_def_id,
                trait_pred.skip_binder().self_ty(),
                |impl_def_id| {
                    trait_impls.push(impl_def_id);
                },
            );
            trait_impls
        };

        let required_trait_path = self.tcx.def_path_str(trait_pred.def_id());
        let traits_with_same_path: UnordSet<_> = self
            .tcx
            .visible_traits()
            .filter(|trait_def_id| *trait_def_id != trait_pred.def_id())
            .map(|trait_def_id| (self.tcx.def_path_str(trait_def_id), trait_def_id))
            .filter(|(p, _)| *p == required_trait_path)
            .collect();

        let traits_with_same_path =
            traits_with_same_path.into_items().into_sorted_stable_ord_by_key(|(p, _)| p);
        let mut suggested = false;
        for (_, trait_with_same_path) in traits_with_same_path {
            let trait_impls = get_trait_impls(trait_with_same_path);
            if trait_impls.is_empty() {
                continue;
            }
            let impl_spans: Vec<_> =
                trait_impls.iter().map(|impl_def_id| self.tcx.def_span(*impl_def_id)).collect();
            err.span_help(
                impl_spans,
                format!("trait impl{} with same name found", pluralize!(trait_impls.len())),
            );
            let trait_crate = self.tcx.crate_name(trait_with_same_path.krate);
            let crate_msg =
                format!("perhaps two different versions of crate `{trait_crate}` are being used?");
            err.note(crate_msg);
            suggested = true;
        }
        suggested
    }

    /// Creates a `PredicateObligation` with `new_self_ty` replacing the existing type in the
    /// `trait_ref`.
    ///
    /// For this to work, `new_self_ty` must have no escaping bound variables.
    pub(super) fn mk_trait_obligation_with_new_self_ty(
        &self,
        param_env: ty::ParamEnv<'tcx>,
        trait_ref_and_ty: ty::Binder<'tcx, (ty::TraitPredicate<'tcx>, Ty<'tcx>)>,
    ) -> PredicateObligation<'tcx> {
        let trait_pred =
            trait_ref_and_ty.map_bound(|(tr, new_self_ty)| tr.with_self_ty(self.tcx, new_self_ty));

        Obligation::new(self.tcx, ObligationCause::dummy(), param_env, trait_pred)
    }

    /// Returns `true` if the trait predicate may apply for *some* assignment
    /// to the type parameters.
    fn predicate_can_apply(
        &self,
        param_env: ty::ParamEnv<'tcx>,
        pred: ty::PolyTraitPredicate<'tcx>,
    ) -> bool {
        struct ParamToVarFolder<'a, 'tcx> {
            infcx: &'a InferCtxt<'tcx>,
            var_map: FxHashMap<Ty<'tcx>, Ty<'tcx>>,
        }

        impl<'a, 'tcx> TypeFolder<TyCtxt<'tcx>> for ParamToVarFolder<'a, 'tcx> {
            fn cx(&self) -> TyCtxt<'tcx> {
                self.infcx.tcx
            }

            fn fold_ty(&mut self, ty: Ty<'tcx>) -> Ty<'tcx> {
                if let ty::Param(_) = *ty.kind() {
                    let infcx = self.infcx;
                    *self.var_map.entry(ty).or_insert_with(|| infcx.next_ty_var(DUMMY_SP))
                } else {
                    ty.super_fold_with(self)
                }
            }
        }

        self.probe(|_| {
            let cleaned_pred =
                pred.fold_with(&mut ParamToVarFolder { infcx: self, var_map: Default::default() });

            let InferOk { value: cleaned_pred, .. } =
                self.infcx.at(&ObligationCause::dummy(), param_env).normalize(cleaned_pred);

            let obligation =
                Obligation::new(self.tcx, ObligationCause::dummy(), param_env, cleaned_pred);

            self.predicate_may_hold(&obligation)
        })
    }

    pub fn note_obligation_cause(
        &self,
        err: &mut Diag<'_>,
        obligation: &PredicateObligation<'tcx>,
    ) {
        // First, attempt to add note to this error with an async-await-specific
        // message, and fall back to regular note otherwise.
        if !self.maybe_note_obligation_cause_for_async_await(err, obligation) {
            self.note_obligation_cause_code(
                obligation.cause.body_id,
                err,
                obligation.predicate,
                obligation.param_env,
                obligation.cause.code(),
                &mut vec![],
                &mut Default::default(),
            );
            self.suggest_swapping_lhs_and_rhs(
                err,
                obligation.predicate,
                obligation.param_env,
                obligation.cause.code(),
            );
            self.suggest_unsized_bound_if_applicable(err, obligation);
            if let Some(span) = err.span.primary_span()
                && let Some(mut diag) =
                    self.dcx().steal_non_err(span, StashKey::AssociatedTypeSuggestion)
                && let Suggestions::Enabled(ref mut s1) = err.suggestions
                && let Suggestions::Enabled(ref mut s2) = diag.suggestions
            {
                s1.append(s2);
                diag.cancel()
            }
        }
    }

    pub(super) fn is_recursive_obligation(
        &self,
        obligated_types: &mut Vec<Ty<'tcx>>,
        cause_code: &ObligationCauseCode<'tcx>,
    ) -> bool {
        if let ObligationCauseCode::BuiltinDerived(data) = cause_code {
            let parent_trait_ref = self.resolve_vars_if_possible(data.parent_trait_pred);
            let self_ty = parent_trait_ref.skip_binder().self_ty();
            if obligated_types.iter().any(|ot| ot == &self_ty) {
                return true;
            }
            if let ty::Adt(def, args) = self_ty.kind()
                && let [arg] = &args[..]
                && let ty::GenericArgKind::Type(ty) = arg.kind()
                && let ty::Adt(inner_def, _) = ty.kind()
                && inner_def == def
            {
                return true;
            }
        }
        false
    }

    fn get_standard_error_message(
        &self,
        trait_predicate: ty::PolyTraitPredicate<'tcx>,
        message: Option<String>,
        predicate_constness: Option<ty::BoundConstness>,
        append_const_msg: Option<AppendConstMessage>,
        post_message: String,
        long_ty_file: &mut Option<PathBuf>,
    ) -> String {
        message
            .and_then(|cannot_do_this| {
                match (predicate_constness, append_const_msg) {
                    // do nothing if predicate is not const
                    (None, _) => Some(cannot_do_this),
                    // suggested using default post message
                    (
                        Some(ty::BoundConstness::Const | ty::BoundConstness::Maybe),
                        Some(AppendConstMessage::Default),
                    ) => Some(format!("{cannot_do_this} in const contexts")),
                    // overridden post message
                    (
                        Some(ty::BoundConstness::Const | ty::BoundConstness::Maybe),
                        Some(AppendConstMessage::Custom(custom_msg, _)),
                    ) => Some(format!("{cannot_do_this}{custom_msg}")),
                    // fallback to generic message
                    (Some(ty::BoundConstness::Const | ty::BoundConstness::Maybe), None) => None,
                }
            })
            .unwrap_or_else(|| {
                format!(
                    "the trait bound `{}` is not satisfied{post_message}",
                    self.tcx.short_string(
                        trait_predicate.print_with_bound_constness(predicate_constness),
                        long_ty_file,
                    ),
                )
            })
    }

    fn get_safe_transmute_error_and_reason(
        &self,
        obligation: PredicateObligation<'tcx>,
        trait_pred: ty::PolyTraitPredicate<'tcx>,
        span: Span,
    ) -> GetSafeTransmuteErrorAndReason {
        use rustc_transmute::Answer;
        self.probe(|_| {
            // We don't assemble a transmutability candidate for types that are generic
            // and we should have ambiguity for types that still have non-region infer.
            if obligation.predicate.has_non_region_param() || obligation.has_non_region_infer() {
                return GetSafeTransmuteErrorAndReason::Default;
            }

            // Erase regions because layout code doesn't particularly care about regions.
            let trait_pred =
                self.tcx.erase_regions(self.tcx.instantiate_bound_regions_with_erased(trait_pred));

            let src_and_dst = rustc_transmute::Types {
                dst: trait_pred.trait_ref.args.type_at(0),
                src: trait_pred.trait_ref.args.type_at(1),
            };

            let ocx = ObligationCtxt::new(self);
            let Ok(assume) = ocx.structurally_normalize_const(
                &obligation.cause,
                obligation.param_env,
                trait_pred.trait_ref.args.const_at(2),
            ) else {
                self.dcx().span_delayed_bug(
                    span,
                    "Unable to construct rustc_transmute::Assume where it was previously possible",
                );
                return GetSafeTransmuteErrorAndReason::Silent;
            };

            let Some(assume) = rustc_transmute::Assume::from_const(self.infcx.tcx, assume) else {
                self.dcx().span_delayed_bug(
                    span,
                    "Unable to construct rustc_transmute::Assume where it was previously possible",
                );
                return GetSafeTransmuteErrorAndReason::Silent;
            };

            let dst = trait_pred.trait_ref.args.type_at(0);
            let src = trait_pred.trait_ref.args.type_at(1);
            let err_msg = format!("`{src}` cannot be safely transmuted into `{dst}`");

            match rustc_transmute::TransmuteTypeEnv::new(self.infcx.tcx)
                .is_transmutable(src_and_dst, assume)
            {
                Answer::No(reason) => {
                    let safe_transmute_explanation = match reason {
                        rustc_transmute::Reason::SrcIsNotYetSupported => {
                            format!("analyzing the transmutability of `{src}` is not yet supported")
                        }

                        rustc_transmute::Reason::DstIsNotYetSupported => {
                            format!("analyzing the transmutability of `{dst}` is not yet supported")
                        }

                        rustc_transmute::Reason::DstIsBitIncompatible => {
                            format!(
                                "at least one value of `{src}` isn't a bit-valid value of `{dst}`"
                            )
                        }

                        rustc_transmute::Reason::DstUninhabited => {
                            format!("`{dst}` is uninhabited")
                        }

                        rustc_transmute::Reason::DstMayHaveSafetyInvariants => {
                            format!("`{dst}` may carry safety invariants")
                        }
                        rustc_transmute::Reason::DstIsTooBig => {
                            format!("the size of `{src}` is smaller than the size of `{dst}`")
                        }
                        rustc_transmute::Reason::DstRefIsTooBig { src, dst } => {
                            let src_size = src.size;
                            let dst_size = dst.size;
                            format!(
                                "the referent size of `{src}` ({src_size} bytes) \
                        is smaller than that of `{dst}` ({dst_size} bytes)"
                            )
                        }
                        rustc_transmute::Reason::SrcSizeOverflow => {
                            format!(
                                "values of the type `{src}` are too big for the target architecture"
                            )
                        }
                        rustc_transmute::Reason::DstSizeOverflow => {
                            format!(
                                "values of the type `{dst}` are too big for the target architecture"
                            )
                        }
                        rustc_transmute::Reason::DstHasStricterAlignment {
                            src_min_align,
                            dst_min_align,
                        } => {
                            format!(
                                "the minimum alignment of `{src}` ({src_min_align}) should \
                        be greater than that of `{dst}` ({dst_min_align})"
                            )
                        }
                        rustc_transmute::Reason::DstIsMoreUnique => {
                            format!(
                                "`{src}` is a shared reference, but `{dst}` is a unique reference"
                            )
                        }
                        // Already reported by rustc
                        rustc_transmute::Reason::TypeError => {
                            return GetSafeTransmuteErrorAndReason::Silent;
                        }
                        rustc_transmute::Reason::SrcLayoutUnknown => {
                            format!("`{src}` has an unknown layout")
                        }
                        rustc_transmute::Reason::DstLayoutUnknown => {
                            format!("`{dst}` has an unknown layout")
                        }
                    };
                    GetSafeTransmuteErrorAndReason::Error {
                        err_msg,
                        safe_transmute_explanation: Some(safe_transmute_explanation),
                    }
                }
                // Should never get a Yes at this point! We already ran it before, and did not get a Yes.
                Answer::Yes => span_bug!(
                    span,
                    "Inconsistent rustc_transmute::is_transmutable(...) result, got Yes",
                ),
                // Reached when a different obligation (namely `Freeze`) causes the
                // transmutability analysis to fail. In this case, silence the
                // transmutability error message in favor of that more specific
                // error.
                Answer::If(_) => GetSafeTransmuteErrorAndReason::Error {
                    err_msg,
                    safe_transmute_explanation: None,
                },
            }
        })
    }

    fn add_tuple_trait_message(
        &self,
        obligation_cause_code: &ObligationCauseCode<'tcx>,
        err: &mut Diag<'_>,
    ) {
        match obligation_cause_code {
            ObligationCauseCode::RustCall => {
                err.primary_message("functions with the \"rust-call\" ABI must take a single non-self tuple argument");
            }
            ObligationCauseCode::WhereClause(def_id, _) if self.tcx.is_fn_trait(*def_id) => {
                err.code(E0059);
                err.primary_message(format!(
                    "type parameter to bare `{}` trait must be a tuple",
                    self.tcx.def_path_str(*def_id)
                ));
            }
            _ => {}
        }
    }

    fn try_to_add_help_message(
        &self,
        root_obligation: &PredicateObligation<'tcx>,
        obligation: &PredicateObligation<'tcx>,
        trait_predicate: ty::PolyTraitPredicate<'tcx>,
        err: &mut Diag<'_>,
        span: Span,
        is_fn_trait: bool,
        suggested: bool,
        unsatisfied_const: bool,
    ) {
        let body_def_id = obligation.cause.body_id;
        let span = if let ObligationCauseCode::BinOp { rhs_span: Some(rhs_span), .. } =
            obligation.cause.code()
        {
            *rhs_span
        } else {
            span
        };

        // Try to report a help message
        let trait_def_id = trait_predicate.def_id();
        if is_fn_trait
            && let Ok((implemented_kind, params)) = self.type_implements_fn_trait(
                obligation.param_env,
                trait_predicate.self_ty(),
                trait_predicate.skip_binder().polarity,
            )
        {
            self.add_help_message_for_fn_trait(trait_predicate, err, implemented_kind, params);
        } else if !trait_predicate.has_non_region_infer()
            && self.predicate_can_apply(obligation.param_env, trait_predicate)
        {
            // If a where-clause may be useful, remind the
            // user that they can add it.
            //
            // don't display an on-unimplemented note, as
            // these notes will often be of the form
            //     "the type `T` can't be frobnicated"
            // which is somewhat confusing.
            self.suggest_restricting_param_bound(
                err,
                trait_predicate,
                None,
                obligation.cause.body_id,
            );
        } else if trait_def_id.is_local()
            && self.tcx.trait_impls_of(trait_def_id).is_empty()
            && !self.tcx.trait_is_auto(trait_def_id)
            && !self.tcx.trait_is_alias(trait_def_id)
            && trait_predicate.polarity() == ty::PredicatePolarity::Positive
        {
            err.span_help(
                self.tcx.def_span(trait_def_id),
                crate::fluent_generated::trait_selection_trait_has_no_impls,
            );
        } else if !suggested
            && !unsatisfied_const
            && trait_predicate.polarity() == ty::PredicatePolarity::Positive
        {
            // Can't show anything else useful, try to find similar impls.
            let impl_candidates = self.find_similar_impl_candidates(trait_predicate);
            if !self.report_similar_impl_candidates(
                &impl_candidates,
                trait_predicate,
                body_def_id,
                err,
                true,
                obligation.param_env,
            ) {
                self.report_similar_impl_candidates_for_root_obligation(
                    obligation,
                    trait_predicate,
                    body_def_id,
                    err,
                );
            }

            self.suggest_convert_to_slice(
                err,
                obligation,
                trait_predicate,
                impl_candidates.as_slice(),
                span,
            );

            self.suggest_tuple_wrapping(err, root_obligation, obligation);
        }
    }

    fn add_help_message_for_fn_trait(
        &self,
        trait_pred: ty::PolyTraitPredicate<'tcx>,
        err: &mut Diag<'_>,
        implemented_kind: ty::ClosureKind,
        params: ty::Binder<'tcx, Ty<'tcx>>,
    ) {
        // If the type implements `Fn`, `FnMut`, or `FnOnce`, suppress the following
        // suggestion to add trait bounds for the type, since we only typically implement
        // these traits once.

        // Note if the `FnMut` or `FnOnce` is less general than the trait we're trying
        // to implement.
        let selected_kind = self
            .tcx
            .fn_trait_kind_from_def_id(trait_pred.def_id())
            .expect("expected to map DefId to ClosureKind");
        if !implemented_kind.extends(selected_kind) {
            err.note(format!(
                "`{}` implements `{}`, but it must implement `{}`, which is more general",
                trait_pred.skip_binder().self_ty(),
                implemented_kind,
                selected_kind
            ));
        }

        // Note any argument mismatches
        let ty::Tuple(given) = *params.skip_binder().kind() else {
            return;
        };

        let expected_ty = trait_pred.skip_binder().trait_ref.args.type_at(1);
        let ty::Tuple(expected) = *expected_ty.kind() else {
            return;
        };

        if expected.len() != given.len() {
            // Note number of types that were expected and given
            err.note(format!(
                "expected a closure taking {} argument{}, but one taking {} argument{} was given",
                given.len(),
                pluralize!(given.len()),
                expected.len(),
                pluralize!(expected.len()),
            ));
            return;
        }

        let given_ty = Ty::new_fn_ptr(
            self.tcx,
            params.rebind(self.tcx.mk_fn_sig(
                given,
                self.tcx.types.unit,
                false,
                hir::Safety::Safe,
                ExternAbi::Rust,
            )),
        );
        let expected_ty = Ty::new_fn_ptr(
            self.tcx,
            trait_pred.rebind(self.tcx.mk_fn_sig(
                expected,
                self.tcx.types.unit,
                false,
                hir::Safety::Safe,
                ExternAbi::Rust,
            )),
        );

        if !self.same_type_modulo_infer(given_ty, expected_ty) {
            // Print type mismatch
            let (expected_args, given_args) = self.cmp(expected_ty, given_ty);
            err.note_expected_found(
                "a closure with signature",
                expected_args,
                "a closure with signature",
                given_args,
            );
        }
    }

    fn maybe_add_note_for_unsatisfied_const(
        &self,
        _trait_predicate: ty::PolyTraitPredicate<'tcx>,
        _err: &mut Diag<'_>,
        _span: Span,
    ) -> UnsatisfiedConst {
        let unsatisfied_const = UnsatisfiedConst(false);
        // FIXME(const_trait_impl)
        unsatisfied_const
    }

    fn report_closure_error(
        &self,
        obligation: &PredicateObligation<'tcx>,
        closure_def_id: DefId,
        found_kind: ty::ClosureKind,
        kind: ty::ClosureKind,
        trait_prefix: &'static str,
    ) -> Diag<'a> {
        let closure_span = self.tcx.def_span(closure_def_id);

        let mut err = ClosureKindMismatch {
            closure_span,
            expected: kind,
            found: found_kind,
            cause_span: obligation.cause.span,
            trait_prefix,
            fn_once_label: None,
            fn_mut_label: None,
        };

        // Additional context information explaining why the closure only implements
        // a particular trait.
        if let Some(typeck_results) = &self.typeck_results {
            let hir_id = self.tcx.local_def_id_to_hir_id(closure_def_id.expect_local());
            match (found_kind, typeck_results.closure_kind_origins().get(hir_id)) {
                (ty::ClosureKind::FnOnce, Some((span, place))) => {
                    err.fn_once_label = Some(ClosureFnOnceLabel {
                        span: *span,
                        place: ty::place_to_string_for_capture(self.tcx, place),
                    })
                }
                (ty::ClosureKind::FnMut, Some((span, place))) => {
                    err.fn_mut_label = Some(ClosureFnMutLabel {
                        span: *span,
                        place: ty::place_to_string_for_capture(self.tcx, place),
                    })
                }
                _ => {}
            }
        }

        self.dcx().create_err(err)
    }

    fn report_cyclic_signature_error(
        &self,
        obligation: &PredicateObligation<'tcx>,
        found_trait_ref: ty::TraitRef<'tcx>,
        expected_trait_ref: ty::TraitRef<'tcx>,
        terr: TypeError<'tcx>,
    ) -> Diag<'a> {
        let self_ty = found_trait_ref.self_ty();
        let (cause, terr) = if let ty::Closure(def_id, _) = self_ty.kind() {
            (
                ObligationCause::dummy_with_span(self.tcx.def_span(def_id)),
                TypeError::CyclicTy(self_ty),
            )
        } else {
            (obligation.cause.clone(), terr)
        };
        self.report_and_explain_type_error(
            TypeTrace::trait_refs(&cause, expected_trait_ref, found_trait_ref),
            obligation.param_env,
            terr,
        )
    }

    fn report_opaque_type_auto_trait_leakage(
        &self,
        obligation: &PredicateObligation<'tcx>,
        def_id: DefId,
    ) -> ErrorGuaranteed {
        let name = match self.tcx.local_opaque_ty_origin(def_id.expect_local()) {
            hir::OpaqueTyOrigin::FnReturn { .. } | hir::OpaqueTyOrigin::AsyncFn { .. } => {
                "opaque type".to_string()
            }
            hir::OpaqueTyOrigin::TyAlias { .. } => {
                format!("`{}`", self.tcx.def_path_debug_str(def_id))
            }
        };
        let mut err = self.dcx().struct_span_err(
            obligation.cause.span,
            format!("cannot check whether the hidden type of {name} satisfies auto traits"),
        );

        err.note(
            "fetching the hidden types of an opaque inside of the defining scope is not supported. \
            You can try moving the opaque type and the item that actually registers a hidden type into a new submodule",
        );
        err.span_note(self.tcx.def_span(def_id), "opaque type is declared here");

        self.note_obligation_cause(&mut err, &obligation);
        self.dcx().try_steal_replace_and_emit_err(self.tcx.def_span(def_id), StashKey::Cycle, err)
    }

    fn report_signature_mismatch_error(
        &self,
        obligation: &PredicateObligation<'tcx>,
        span: Span,
        found_trait_ref: ty::TraitRef<'tcx>,
        expected_trait_ref: ty::TraitRef<'tcx>,
    ) -> Result<Diag<'a>, ErrorGuaranteed> {
        let found_trait_ref = self.resolve_vars_if_possible(found_trait_ref);
        let expected_trait_ref = self.resolve_vars_if_possible(expected_trait_ref);

        expected_trait_ref.self_ty().error_reported()?;
        let found_trait_ty = found_trait_ref.self_ty();

        let found_did = match *found_trait_ty.kind() {
            ty::Closure(did, _) | ty::FnDef(did, _) | ty::Coroutine(did, ..) => Some(did),
            _ => None,
        };

        let found_node = found_did.and_then(|did| self.tcx.hir_get_if_local(did));
        let found_span = found_did.and_then(|did| self.tcx.hir_span_if_local(did));

        if !self.reported_signature_mismatch.borrow_mut().insert((span, found_span)) {
            // We check closures twice, with obligations flowing in different directions,
            // but we want to complain about them only once.
            return Err(self.dcx().span_delayed_bug(span, "already_reported"));
        }

        let mut not_tupled = false;

        let found = match found_trait_ref.args.type_at(1).kind() {
            ty::Tuple(tys) => vec![ArgKind::empty(); tys.len()],
            _ => {
                not_tupled = true;
                vec![ArgKind::empty()]
            }
        };

        let expected_ty = expected_trait_ref.args.type_at(1);
        let expected = match expected_ty.kind() {
            ty::Tuple(tys) => {
                tys.iter().map(|t| ArgKind::from_expected_ty(t, Some(span))).collect()
            }
            _ => {
                not_tupled = true;
                vec![ArgKind::Arg("_".to_owned(), expected_ty.to_string())]
            }
        };

        // If this is a `Fn` family trait and either the expected or found
        // is not tupled, then fall back to just a regular mismatch error.
        // This shouldn't be common unless manually implementing one of the
        // traits manually, but don't make it more confusing when it does
        // happen.
        if !self.tcx.is_lang_item(expected_trait_ref.def_id, LangItem::Coroutine) && not_tupled {
            return Ok(self.report_and_explain_type_error(
                TypeTrace::trait_refs(&obligation.cause, expected_trait_ref, found_trait_ref),
                obligation.param_env,
                ty::error::TypeError::Mismatch,
            ));
        }
        if found.len() != expected.len() {
            let (closure_span, closure_arg_span, found) = found_did
                .and_then(|did| {
                    let node = self.tcx.hir_get_if_local(did)?;
                    let (found_span, closure_arg_span, found) = self.get_fn_like_arguments(node)?;
                    Some((Some(found_span), closure_arg_span, found))
                })
                .unwrap_or((found_span, None, found));

            // If the coroutine take a single () as its argument,
            // the trait argument would found the coroutine take 0 arguments,
            // but get_fn_like_arguments would give 1 argument.
            // This would result in "Expected to take 1 argument, but it takes 1 argument".
            // Check again to avoid this.
            if found.len() != expected.len() {
                return Ok(self.report_arg_count_mismatch(
                    span,
                    closure_span,
                    expected,
                    found,
                    found_trait_ty.is_closure(),
                    closure_arg_span,
                ));
            }
        }
        Ok(self.report_closure_arg_mismatch(
            span,
            found_span,
            found_trait_ref,
            expected_trait_ref,
            obligation.cause.code(),
            found_node,
            obligation.param_env,
        ))
    }

    /// Given some node representing a fn-like thing in the HIR map,
    /// returns a span and `ArgKind` information that describes the
    /// arguments it expects. This can be supplied to
    /// `report_arg_count_mismatch`.
    pub fn get_fn_like_arguments(
        &self,
        node: Node<'_>,
    ) -> Option<(Span, Option<Span>, Vec<ArgKind>)> {
        let sm = self.tcx.sess.source_map();
        Some(match node {
            Node::Expr(&hir::Expr {
                kind: hir::ExprKind::Closure(&hir::Closure { body, fn_decl_span, fn_arg_span, .. }),
                ..
            }) => (
                fn_decl_span,
                fn_arg_span,
                self.tcx
                    .hir_body(body)
                    .params
                    .iter()
                    .map(|arg| {
                        if let hir::Pat { kind: hir::PatKind::Tuple(args, _), span, .. } = *arg.pat
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
            Node::Item(&hir::Item { kind: hir::ItemKind::Fn { ref sig, .. }, .. })
            | Node::ImplItem(&hir::ImplItem { kind: hir::ImplItemKind::Fn(ref sig, _), .. })
            | Node::TraitItem(&hir::TraitItem {
                kind: hir::TraitItemKind::Fn(ref sig, _), ..
            })
            | Node::ForeignItem(&hir::ForeignItem {
                kind: hir::ForeignItemKind::Fn(ref sig, _, _),
                ..
            }) => (
                sig.span,
                None,
                sig.decl
                    .inputs
                    .iter()
                    .map(|arg| match arg.kind {
                        hir::TyKind::Tup(tys) => ArgKind::Tuple(
                            Some(arg.span),
                            vec![("_".to_owned(), "_".to_owned()); tys.len()],
                        ),
                        _ => ArgKind::empty(),
                    })
                    .collect::<Vec<ArgKind>>(),
            ),
            Node::Ctor(variant_data) => {
                let span = variant_data.ctor_hir_id().map_or(DUMMY_SP, |id| self.tcx.hir_span(id));
                (span, None, vec![ArgKind::empty(); variant_data.fields().len()])
            }
            _ => panic!("non-FnLike node found: {node:?}"),
        })
    }

    /// Reports an error when the number of arguments needed by a
    /// trait match doesn't match the number that the expression
    /// provides.
    pub fn report_arg_count_mismatch(
        &self,
        span: Span,
        found_span: Option<Span>,
        expected_args: Vec<ArgKind>,
        found_args: Vec<ArgKind>,
        is_closure: bool,
        closure_arg_span: Option<Span>,
    ) -> Diag<'a> {
        let kind = if is_closure { "closure" } else { "function" };

        let args_str = |arguments: &[ArgKind], other: &[ArgKind]| {
            let arg_length = arguments.len();
            let distinct = matches!(other, &[ArgKind::Tuple(..)]);
            match (arg_length, arguments.get(0)) {
                (1, Some(ArgKind::Tuple(_, fields))) => {
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

        let mut err = struct_span_code_err!(
            self.dcx(),
            span,
            E0593,
            "{} is expected to take {}, but it takes {}",
            kind,
            expected_str,
            found_str,
        );

        err.span_label(span, format!("expected {kind} that takes {expected_str}"));

        if let Some(found_span) = found_span {
            err.span_label(found_span, format!("takes {found_str}"));

            // Suggest to take and ignore the arguments with expected_args_length `_`s if
            // found arguments is empty (assume the user just wants to ignore args in this case).
            // For example, if `expected_args_length` is 2, suggest `|_, _|`.
            if found_args.is_empty() && is_closure {
                let underscores = vec!["_"; expected_args.len()].join(", ");
                err.span_suggestion_verbose(
                    closure_arg_span.unwrap_or(found_span),
                    format!(
                        "consider changing the closure to take and ignore the expected argument{}",
                        pluralize!(expected_args.len())
                    ),
                    format!("|{underscores}|"),
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
                        format!("|{sugg}|"),
                        Applicability::MachineApplicable,
                    );
                }
            }
            if let &[ArgKind::Tuple(_, ref fields)] = &expected_args[..]
                && fields.len() == found_args.len()
                && is_closure
            {
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

        err
    }

    /// Checks if the type implements one of `Fn`, `FnMut`, or `FnOnce`
    /// in that order, and returns the generic type corresponding to the
    /// argument of that trait (corresponding to the closure arguments).
    pub fn type_implements_fn_trait(
        &self,
        param_env: ty::ParamEnv<'tcx>,
        ty: ty::Binder<'tcx, Ty<'tcx>>,
        polarity: ty::PredicatePolarity,
    ) -> Result<(ty::ClosureKind, ty::Binder<'tcx, Ty<'tcx>>), ()> {
        self.commit_if_ok(|_| {
            for trait_def_id in [
                self.tcx.lang_items().fn_trait(),
                self.tcx.lang_items().fn_mut_trait(),
                self.tcx.lang_items().fn_once_trait(),
            ] {
                let Some(trait_def_id) = trait_def_id else { continue };
                // Make a fresh inference variable so we can determine what the generic parameters
                // of the trait are.
                let var = self.next_ty_var(DUMMY_SP);
                // FIXME(const_trait_impl)
                let trait_ref = ty::TraitRef::new(self.tcx, trait_def_id, [ty.skip_binder(), var]);
                let obligation = Obligation::new(
                    self.tcx,
                    ObligationCause::dummy(),
                    param_env,
                    ty.rebind(ty::TraitPredicate { trait_ref, polarity }),
                );
                let ocx = ObligationCtxt::new(self);
                ocx.register_obligation(obligation);
                if ocx.select_all_or_error().is_empty() {
                    return Ok((
                        self.tcx
                            .fn_trait_kind_from_def_id(trait_def_id)
                            .expect("expected to map DefId to ClosureKind"),
                        ty.rebind(self.resolve_vars_if_possible(var)),
                    ));
                }
            }

            Err(())
        })
    }

    fn report_not_const_evaluatable_error(
        &self,
        obligation: &PredicateObligation<'tcx>,
        span: Span,
    ) -> Result<Diag<'a>, ErrorGuaranteed> {
        if !self.tcx.features().generic_const_exprs()
            && !self.tcx.features().min_generic_const_args()
        {
            let guar = self
                .dcx()
                .struct_span_err(span, "constant expression depends on a generic parameter")
                // FIXME(const_generics): we should suggest to the user how they can resolve this
                // issue. However, this is currently not actually possible
                // (see https://github.com/rust-lang/rust/issues/66962#issuecomment-575907083).
                //
                // Note that with `feature(generic_const_exprs)` this case should not
                // be reachable.
                .with_note("this may fail depending on what value the parameter takes")
                .emit();
            return Err(guar);
        }

        match obligation.predicate.kind().skip_binder() {
            ty::PredicateKind::Clause(ty::ClauseKind::ConstEvaluatable(ct)) => match ct.kind() {
                ty::ConstKind::Unevaluated(uv) => {
                    let mut err =
                        self.dcx().struct_span_err(span, "unconstrained generic constant");
                    let const_span = self.tcx.def_span(uv.def);

                    let const_ty = self.tcx.type_of(uv.def).instantiate(self.tcx, uv.args);
                    let cast = if const_ty != self.tcx.types.usize { " as usize" } else { "" };
                    let msg = "try adding a `where` bound";
                    match self.tcx.sess.source_map().span_to_snippet(const_span) {
                        Ok(snippet) => {
                            let code = format!("[(); {snippet}{cast}]:");
                            let def_id = if let ObligationCauseCode::CompareImplItem {
                                trait_item_def_id,
                                ..
                            } = obligation.cause.code()
                            {
                                trait_item_def_id.as_local()
                            } else {
                                Some(obligation.cause.body_id)
                            };
                            if let Some(def_id) = def_id
                                && let Some(generics) = self.tcx.hir_get_generics(def_id)
                            {
                                err.span_suggestion_verbose(
                                    generics.tail_span_for_predicate_suggestion(),
                                    msg,
                                    format!("{} {code}", generics.add_where_or_trailing_comma()),
                                    Applicability::MaybeIncorrect,
                                );
                            } else {
                                err.help(format!("{msg}: where {code}"));
                            };
                        }
                        _ => {
                            err.help(msg);
                        }
                    };
                    Ok(err)
                }
                ty::ConstKind::Expr(_) => {
                    let err = self
                        .dcx()
                        .struct_span_err(span, format!("unconstrained generic constant `{ct}`"));
                    Ok(err)
                }
                _ => {
                    bug!("const evaluatable failed for non-unevaluated const `{ct:?}`");
                }
            },
            _ => {
                span_bug!(
                    span,
                    "unexpected non-ConstEvaluatable predicate, this should not be reachable"
                )
            }
        }
    }
}
