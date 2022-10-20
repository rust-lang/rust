//! Errors emitted by `rustc_hir_analysis`.
use rustc_macros::{Diagnostic, Subdiagnostic};
use rustc_middle::ty::Ty;
use rustc_span::{symbol::Ident, Span};

#[derive(Diagnostic)]
#[diag(hir_analysis::field_multiply_specified_in_initializer, code = "E0062")]
pub struct FieldMultiplySpecifiedInInitializer {
    #[primary_span]
    #[label]
    pub span: Span,
    #[label(hir_analysis::previous_use_label)]
    pub prev_span: Span,
    pub ident: Ident,
}

#[derive(Diagnostic)]
#[diag(hir_analysis::return_stmt_outside_of_fn_body, code = "E0572")]
pub struct ReturnStmtOutsideOfFnBody {
    #[primary_span]
    pub span: Span,
    #[label(hir_analysis::encl_body_label)]
    pub encl_body_span: Option<Span>,
    #[label(hir_analysis::encl_fn_label)]
    pub encl_fn_span: Option<Span>,
}

#[derive(Diagnostic)]
#[diag(hir_analysis::yield_expr_outside_of_generator, code = "E0627")]
pub struct YieldExprOutsideOfGenerator {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(hir_analysis::struct_expr_non_exhaustive, code = "E0639")]
pub struct StructExprNonExhaustive {
    #[primary_span]
    pub span: Span,
    pub what: &'static str,
}

#[derive(Diagnostic)]
#[diag(hir_analysis::method_call_on_unknown_type, code = "E0699")]
pub struct MethodCallOnUnknownType {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(hir_analysis::functional_record_update_on_non_struct, code = "E0436")]
pub struct FunctionalRecordUpdateOnNonStruct {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(hir_analysis::address_of_temporary_taken, code = "E0745")]
pub struct AddressOfTemporaryTaken {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(Subdiagnostic)]
pub enum AddReturnTypeSuggestion {
    #[suggestion(
        hir_analysis::add_return_type_add,
        code = "-> {found} ",
        applicability = "machine-applicable"
    )]
    Add {
        #[primary_span]
        span: Span,
        found: String,
    },
    #[suggestion(
        hir_analysis::add_return_type_missing_here,
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
    #[label(hir_analysis::expected_default_return_type)]
    Unit {
        #[primary_span]
        span: Span,
    },
    #[label(hir_analysis::expected_return_type)]
    Other {
        #[primary_span]
        span: Span,
        expected: Ty<'tcx>,
    },
}

#[derive(Diagnostic)]
#[diag(hir_analysis::missing_parentheses_in_range, code = "E0689")]
pub struct MissingParentheseInRange {
    #[primary_span]
    #[label(hir_analysis::missing_parentheses_in_range)]
    pub span: Span,
    pub ty_str: String,
    pub method_name: String,

    #[subdiagnostic]
    pub add_missing_parentheses: Option<AddMissingParenthesesInRange>,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion_verbose(
    hir_analysis::add_missing_parentheses_in_range,
    applicability = "maybe-incorrect"
)]
pub struct AddMissingParenthesesInRange {
    pub func_name: String,
    #[suggestion_part(code = "(")]
    pub left: Span,
    #[suggestion_part(code = ")")]
    pub right: Span,
}
