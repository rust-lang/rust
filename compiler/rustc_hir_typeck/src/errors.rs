//! Errors emitted by `rustc_hir_typeck`.
use std::borrow::Cow;

use crate::fluent_generated as fluent;
use rustc_errors::{AddToDiagnostic, Applicability, Diagnostic, MultiSpan, SubdiagnosticMessage};
use rustc_macros::{Diagnostic, Subdiagnostic};
use rustc_middle::ty::Ty;
use rustc_span::{
    edition::{Edition, LATEST_STABLE_EDITION},
    symbol::Ident,
    Span,
};

#[derive(Diagnostic)]
#[diag(hir_typeck_field_multiply_specified_in_initializer, code = "E0062")]
pub struct FieldMultiplySpecifiedInInitializer {
    #[primary_span]
    #[label]
    pub span: Span,
    #[label(hir_typeck_previous_use_label)]
    pub prev_span: Span,
    pub ident: Ident,
}

#[derive(Diagnostic)]
#[diag(hir_typeck_return_stmt_outside_of_fn_body, code = "E0572")]
pub struct ReturnStmtOutsideOfFnBody {
    #[primary_span]
    pub span: Span,
    #[label(hir_typeck_encl_body_label)]
    pub encl_body_span: Option<Span>,
    #[label(hir_typeck_encl_fn_label)]
    pub encl_fn_span: Option<Span>,
}

#[derive(Diagnostic)]
#[diag(hir_typeck_yield_expr_outside_of_generator, code = "E0627")]
pub struct YieldExprOutsideOfGenerator {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(hir_typeck_struct_expr_non_exhaustive, code = "E0639")]
pub struct StructExprNonExhaustive {
    #[primary_span]
    pub span: Span,
    pub what: &'static str,
}

#[derive(Diagnostic)]
#[diag(hir_typeck_method_call_on_unknown_raw_pointee, code = "E0699")]
pub struct MethodCallOnUnknownRawPointee {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(hir_typeck_functional_record_update_on_non_struct, code = "E0436")]
pub struct FunctionalRecordUpdateOnNonStruct {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(hir_typeck_address_of_temporary_taken, code = "E0745")]
pub struct AddressOfTemporaryTaken {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(Subdiagnostic)]
pub enum AddReturnTypeSuggestion {
    #[suggestion(
        hir_typeck_add_return_type_add,
        code = "-> {found} ",
        applicability = "machine-applicable"
    )]
    Add {
        #[primary_span]
        span: Span,
        found: String,
    },
    #[suggestion(
        hir_typeck_add_return_type_missing_here,
        code = "-> _ ",
        applicability = "has-placeholders"
    )]
    MissingHere {
        #[primary_span]
        span: Span,
    },
}

#[derive(Subdiagnostic)]
pub enum ExpectedReturnTypeLabel<'tcx> {
    #[label(hir_typeck_expected_default_return_type)]
    Unit {
        #[primary_span]
        span: Span,
    },
    #[label(hir_typeck_expected_return_type)]
    Other {
        #[primary_span]
        span: Span,
        expected: Ty<'tcx>,
    },
}

#[derive(Diagnostic)]
#[diag(hir_typeck_missing_parentheses_in_range, code = "E0689")]
pub struct MissingParenthesesInRange {
    #[primary_span]
    #[label(hir_typeck_missing_parentheses_in_range)]
    pub span: Span,
    pub ty_str: String,
    pub method_name: String,
    #[subdiagnostic]
    pub add_missing_parentheses: Option<AddMissingParenthesesInRange>,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    hir_typeck_add_missing_parentheses_in_range,
    style = "verbose",
    applicability = "maybe-incorrect"
)]
pub struct AddMissingParenthesesInRange {
    pub func_name: String,
    #[suggestion_part(code = "(")]
    pub left: Span,
    #[suggestion_part(code = ")")]
    pub right: Span,
}

#[derive(Diagnostic)]
#[diag(hir_typeck_op_trait_generic_params)]
pub struct OpMethodGenericParams {
    #[primary_span]
    pub span: Span,
    pub method_name: String,
}

pub struct TypeMismatchFruTypo {
    /// Span of the LHS of the range
    pub expr_span: Span,
    /// Span of the `..RHS` part of the range
    pub fru_span: Span,
    /// Rendered expression of the RHS of the range
    pub expr: Option<String>,
}

impl AddToDiagnostic for TypeMismatchFruTypo {
    fn add_to_diagnostic_with<F>(self, diag: &mut Diagnostic, _: F)
    where
        F: Fn(&mut Diagnostic, SubdiagnosticMessage) -> SubdiagnosticMessage,
    {
        diag.set_arg("expr", self.expr.as_deref().unwrap_or("NONE"));

        // Only explain that `a ..b` is a range if it's split up
        if self.expr_span.between(self.fru_span).is_empty() {
            diag.span_note(self.expr_span.to(self.fru_span), fluent::hir_typeck_fru_note);
        } else {
            let mut multispan: MultiSpan = vec![self.expr_span, self.fru_span].into();
            multispan.push_span_label(self.expr_span, fluent::hir_typeck_fru_expr);
            multispan.push_span_label(self.fru_span, fluent::hir_typeck_fru_expr2);
            diag.span_note(multispan, fluent::hir_typeck_fru_note);
        }

        diag.span_suggestion(
            self.expr_span.shrink_to_hi(),
            fluent::hir_typeck_fru_suggestion,
            ", ",
            Applicability::MaybeIncorrect,
        );
    }
}

#[derive(Diagnostic)]
#[diag(hir_typeck_lang_start_incorrect_number_params)]
#[note(hir_typeck_lang_start_incorrect_number_params_note_expected_count)]
#[note(hir_typeck_lang_start_expected_sig_note)]
pub struct LangStartIncorrectNumberArgs {
    #[primary_span]
    pub params_span: Span,
    pub found_param_count: usize,
}

#[derive(Diagnostic)]
#[diag(hir_typeck_lang_start_incorrect_param)]
pub struct LangStartIncorrectParam<'tcx> {
    #[primary_span]
    #[suggestion(style = "short", code = "{expected_ty}", applicability = "machine-applicable")]
    pub param_span: Span,

    pub param_num: usize,
    pub expected_ty: Ty<'tcx>,
    pub found_ty: Ty<'tcx>,
}

#[derive(Diagnostic)]
#[diag(hir_typeck_lang_start_incorrect_ret_ty)]
pub struct LangStartIncorrectRetTy<'tcx> {
    #[primary_span]
    #[suggestion(style = "short", code = "{expected_ty}", applicability = "machine-applicable")]
    pub ret_span: Span,

    pub expected_ty: Ty<'tcx>,
    pub found_ty: Ty<'tcx>,
}

#[derive(Subdiagnostic)]
pub enum HelpUseLatestEdition {
    #[help(hir_typeck_help_set_edition_cargo)]
    #[note(hir_typeck_note_edition_guide)]
    Cargo { edition: Edition },
    #[help(hir_typeck_help_set_edition_standalone)]
    #[note(hir_typeck_note_edition_guide)]
    Standalone { edition: Edition },
}

impl HelpUseLatestEdition {
    pub fn new() -> Self {
        let edition = LATEST_STABLE_EDITION;
        if std::env::var_os("CARGO").is_some() {
            Self::Cargo { edition }
        } else {
            Self::Standalone { edition }
        }
    }
}

#[derive(Diagnostic)]
#[diag(hir_typeck_const_select_must_be_const)]
#[help]
pub struct ConstSelectMustBeConst {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(hir_typeck_const_select_must_be_fn)]
#[note]
#[help]
pub struct ConstSelectMustBeFn<'a> {
    #[primary_span]
    pub span: Span,
    pub ty: Ty<'a>,
}

#[derive(Diagnostic)]
#[diag(hir_typeck_union_pat_multiple_fields)]
pub struct UnionPatMultipleFields {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(hir_typeck_union_pat_dotdot)]
pub struct UnionPatDotDot {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(hir_typeck_arg_mismatch_indeterminate)]
pub struct ArgMismatchIndeterminate {
    #[primary_span]
    pub span: Span,
}

#[derive(Subdiagnostic)]
pub enum SuggestBoxing {
    #[note(hir_typeck_suggest_boxing_note)]
    #[multipart_suggestion(
        hir_typeck_suggest_boxing_when_appropriate,
        applicability = "machine-applicable"
    )]
    Unit {
        #[suggestion_part(code = "Box::new(())")]
        start: Span,
        #[suggestion_part(code = "")]
        end: Span,
    },
    #[note(hir_typeck_suggest_boxing_note)]
    AsyncBody,
    #[note(hir_typeck_suggest_boxing_note)]
    #[multipart_suggestion(
        hir_typeck_suggest_boxing_when_appropriate,
        applicability = "machine-applicable"
    )]
    Other {
        #[suggestion_part(code = "Box::new(")]
        start: Span,
        #[suggestion_part(code = ")")]
        end: Span,
    },
}

#[derive(Subdiagnostic)]
#[suggestion(
    hir_typeck_suggest_ptr_null_mut,
    applicability = "maybe-incorrect",
    code = "core::ptr::null_mut()"
)]
pub struct SuggestPtrNullMut {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(hir_typeck_no_associated_item, code = "E0599")]
pub struct NoAssociatedItem {
    #[primary_span]
    pub span: Span,
    pub item_kind: &'static str,
    pub item_name: Ident,
    pub ty_prefix: Cow<'static, str>,
    pub ty_str: String,
    pub trait_missing_method: bool,
}

#[derive(Subdiagnostic)]
#[note(hir_typeck_candidate_trait_note)]
pub struct CandidateTraitNote {
    #[primary_span]
    pub span: Span,
    pub trait_name: String,
    pub item_name: Ident,
    pub action_or_ty: String,
}

#[derive(Diagnostic)]
#[diag(hir_typeck_ctor_is_private, code = "E0603")]
pub struct CtorIsPrivate {
    #[primary_span]
    pub span: Span,
    pub def: String,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    hir_typeck_convert_using_method,
    applicability = "machine-applicable",
    style = "verbose"
)]
pub struct SuggestConvertViaMethod<'tcx> {
    #[suggestion_part(code = "{sugg}")]
    pub span: Span,
    #[suggestion_part(code = "")]
    pub borrow_removal_span: Option<Span>,
    pub sugg: &'static str,
    pub expected: Ty<'tcx>,
    pub found: Ty<'tcx>,
}
