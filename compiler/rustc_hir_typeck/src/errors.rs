//! Errors emitted by `rustc_hir_typeck`.

use std::borrow::Cow;

use rustc_errors::codes::*;
use rustc_errors::{
    Applicability, Diag, DiagArgValue, DiagSymbolList, EmissionGuarantee, IntoDiagArg, MultiSpan,
    SubdiagMessageOp, Subdiagnostic,
};
use rustc_macros::{Diagnostic, LintDiagnostic, Subdiagnostic};
use rustc_middle::ty::{self, Ty};
use rustc_span::edition::{Edition, LATEST_STABLE_EDITION};
use rustc_span::{Ident, Span, Symbol};

use crate::fluent_generated as fluent;

#[derive(Diagnostic)]
#[diag(hir_typeck_base_expression_double_dot, code = E0797)]
pub(crate) struct BaseExpressionDoubleDot {
    #[primary_span]
    pub span: Span,
    #[suggestion(
        hir_typeck_base_expression_double_dot_enable_default_field_values,
        code = "#![feature(default_field_values)]\n",
        applicability = "machine-applicable",
        style = "verbose"
    )]
    pub default_field_values_suggestion: Option<Span>,
    #[subdiagnostic]
    pub default_field_values_help: Option<BaseExpressionDoubleDotEnableDefaultFieldValues>,
    #[subdiagnostic]
    pub add_expr: Option<BaseExpressionDoubleDotAddExpr>,
    #[subdiagnostic]
    pub remove_dots: Option<BaseExpressionDoubleDotRemove>,
}

#[derive(Subdiagnostic)]
#[suggestion(
    hir_typeck_base_expression_double_dot_remove,
    code = "",
    applicability = "machine-applicable",
    style = "verbose"
)]
pub(crate) struct BaseExpressionDoubleDotRemove {
    #[primary_span]
    pub span: Span,
}

#[derive(Subdiagnostic)]
#[suggestion(
    hir_typeck_base_expression_double_dot_add_expr,
    code = "/* expr */",
    applicability = "has-placeholders",
    style = "verbose"
)]
pub(crate) struct BaseExpressionDoubleDotAddExpr {
    #[primary_span]
    pub span: Span,
}

#[derive(Subdiagnostic)]
#[help(hir_typeck_base_expression_double_dot_enable_default_field_values)]
pub(crate) struct BaseExpressionDoubleDotEnableDefaultFieldValues;

#[derive(Diagnostic)]
#[diag(hir_typeck_field_multiply_specified_in_initializer, code = E0062)]
pub(crate) struct FieldMultiplySpecifiedInInitializer {
    #[primary_span]
    #[label]
    pub span: Span,
    #[label(hir_typeck_previous_use_label)]
    pub prev_span: Span,
    pub ident: Ident,
}

#[derive(Diagnostic)]
#[diag(hir_typeck_return_stmt_outside_of_fn_body, code = E0572)]
pub(crate) struct ReturnStmtOutsideOfFnBody {
    #[primary_span]
    pub span: Span,
    #[label(hir_typeck_encl_body_label)]
    pub encl_body_span: Option<Span>,
    #[label(hir_typeck_encl_fn_label)]
    pub encl_fn_span: Option<Span>,
    pub statement_kind: ReturnLikeStatementKind,
}

pub(crate) enum ReturnLikeStatementKind {
    Return,
    Become,
}

impl IntoDiagArg for ReturnLikeStatementKind {
    fn into_diag_arg(self, _: &mut Option<std::path::PathBuf>) -> DiagArgValue {
        let kind = match self {
            Self::Return => "return",
            Self::Become => "become",
        }
        .into();

        DiagArgValue::Str(kind)
    }
}

#[derive(Diagnostic)]
#[diag(hir_typeck_rustcall_incorrect_args)]
pub(crate) struct RustCallIncorrectArgs {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(hir_typeck_yield_expr_outside_of_coroutine, code = E0627)]
pub(crate) struct YieldExprOutsideOfCoroutine {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(hir_typeck_struct_expr_non_exhaustive, code = E0639)]
pub(crate) struct StructExprNonExhaustive {
    #[primary_span]
    pub span: Span,
    pub what: &'static str,
}

#[derive(Diagnostic)]
#[diag(hir_typeck_functional_record_update_on_non_struct, code = E0436)]
pub(crate) struct FunctionalRecordUpdateOnNonStruct {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(hir_typeck_address_of_temporary_taken, code = E0745)]
pub(crate) struct AddressOfTemporaryTaken {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(Subdiagnostic)]
pub(crate) enum AddReturnTypeSuggestion {
    #[suggestion(
        hir_typeck_add_return_type_add,
        code = " -> {found}",
        applicability = "machine-applicable"
    )]
    Add {
        #[primary_span]
        span: Span,
        found: String,
    },
    #[suggestion(
        hir_typeck_add_return_type_missing_here,
        code = " -> _",
        applicability = "has-placeholders"
    )]
    MissingHere {
        #[primary_span]
        span: Span,
    },
}

#[derive(Subdiagnostic)]
pub(crate) enum ExpectedReturnTypeLabel<'tcx> {
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
#[diag(hir_typeck_explicit_destructor, code = E0040)]
pub(crate) struct ExplicitDestructorCall {
    #[primary_span]
    #[label]
    pub span: Span,
    #[subdiagnostic]
    pub sugg: ExplicitDestructorCallSugg,
}

#[derive(Subdiagnostic)]
pub(crate) enum ExplicitDestructorCallSugg {
    #[suggestion(hir_typeck_suggestion, code = "drop", applicability = "maybe-incorrect")]
    Empty(#[primary_span] Span),
    #[multipart_suggestion(hir_typeck_suggestion, style = "short")]
    Snippet {
        #[suggestion_part(code = "drop(")]
        lo: Span,
        #[suggestion_part(code = ")")]
        hi: Span,
    },
}

#[derive(Diagnostic)]
#[diag(hir_typeck_missing_parentheses_in_range, code = E0689)]
pub(crate) struct MissingParenthesesInRange {
    #[primary_span]
    #[label(hir_typeck_missing_parentheses_in_range)]
    pub span: Span,
    pub ty_str: String,
    pub method_name: String,
    #[subdiagnostic]
    pub add_missing_parentheses: Option<AddMissingParenthesesInRange>,
}

#[derive(LintDiagnostic)]
pub(crate) enum NeverTypeFallbackFlowingIntoUnsafe {
    #[help]
    #[diag(hir_typeck_never_type_fallback_flowing_into_unsafe_call)]
    Call {
        #[subdiagnostic]
        sugg: SuggestAnnotations,
    },
    #[help]
    #[diag(hir_typeck_never_type_fallback_flowing_into_unsafe_method)]
    Method {
        #[subdiagnostic]
        sugg: SuggestAnnotations,
    },
    #[help]
    #[diag(hir_typeck_never_type_fallback_flowing_into_unsafe_path)]
    Path {
        #[subdiagnostic]
        sugg: SuggestAnnotations,
    },
    #[help]
    #[diag(hir_typeck_never_type_fallback_flowing_into_unsafe_union_field)]
    UnionField {
        #[subdiagnostic]
        sugg: SuggestAnnotations,
    },
    #[help]
    #[diag(hir_typeck_never_type_fallback_flowing_into_unsafe_deref)]
    Deref {
        #[subdiagnostic]
        sugg: SuggestAnnotations,
    },
}

#[derive(LintDiagnostic)]
#[help]
#[diag(hir_typeck_dependency_on_unit_never_type_fallback)]
pub(crate) struct DependencyOnUnitNeverTypeFallback<'tcx> {
    #[note]
    pub obligation_span: Span,
    pub obligation: ty::Predicate<'tcx>,
    #[subdiagnostic]
    pub sugg: SuggestAnnotations,
}

#[derive(Clone)]
pub(crate) enum SuggestAnnotation {
    Unit(Span),
    Path(Span),
    Local(Span),
    Turbo(Span, usize, usize),
}

#[derive(Clone)]
pub(crate) struct SuggestAnnotations {
    pub suggestions: Vec<SuggestAnnotation>,
}
impl Subdiagnostic for SuggestAnnotations {
    fn add_to_diag_with<G: EmissionGuarantee, F: SubdiagMessageOp<G>>(
        self,
        diag: &mut Diag<'_, G>,
        _: &F,
    ) {
        if self.suggestions.is_empty() {
            return;
        }

        let mut suggestions = vec![];
        for suggestion in self.suggestions {
            match suggestion {
                SuggestAnnotation::Unit(span) => {
                    suggestions.push((span, "()".to_string()));
                }
                SuggestAnnotation::Path(span) => {
                    suggestions.push((span.shrink_to_lo(), "<() as ".to_string()));
                    suggestions.push((span.shrink_to_hi(), ">".to_string()));
                }
                SuggestAnnotation::Local(span) => {
                    suggestions.push((span, ": ()".to_string()));
                }
                SuggestAnnotation::Turbo(span, n_args, idx) => suggestions.push((
                    span,
                    format!(
                        "::<{}>",
                        (0..n_args)
                            .map(|i| if i == idx { "()" } else { "_" })
                            .collect::<Vec<_>>()
                            .join(", "),
                    ),
                )),
            }
        }

        diag.multipart_suggestion_verbose(
            "use `()` annotations to avoid fallback changes",
            suggestions,
            Applicability::MachineApplicable,
        );
    }
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    hir_typeck_add_missing_parentheses_in_range,
    style = "verbose",
    applicability = "maybe-incorrect"
)]
pub(crate) struct AddMissingParenthesesInRange {
    pub func_name: String,
    #[suggestion_part(code = "(")]
    pub left: Span,
    #[suggestion_part(code = ")")]
    pub right: Span,
}

pub(crate) struct TypeMismatchFruTypo {
    /// Span of the LHS of the range
    pub expr_span: Span,
    /// Span of the `..RHS` part of the range
    pub fru_span: Span,
    /// Rendered expression of the RHS of the range
    pub expr: Option<String>,
}

impl Subdiagnostic for TypeMismatchFruTypo {
    fn add_to_diag_with<G: EmissionGuarantee, F: SubdiagMessageOp<G>>(
        self,
        diag: &mut Diag<'_, G>,
        _f: &F,
    ) {
        diag.arg("expr", self.expr.as_deref().unwrap_or("NONE"));

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

#[derive(LintDiagnostic)]
#[diag(hir_typeck_lossy_provenance_int2ptr)]
#[help]
pub(crate) struct LossyProvenanceInt2Ptr<'tcx> {
    pub expr_ty: Ty<'tcx>,
    pub cast_ty: Ty<'tcx>,
    #[subdiagnostic]
    pub sugg: LossyProvenanceInt2PtrSuggestion,
}

#[derive(Diagnostic)]
#[diag(hir_typeck_ptr_cast_add_auto_to_object, code = E0804)]
#[note]
#[help]
pub(crate) struct PtrCastAddAutoToObject {
    #[primary_span]
    #[label]
    pub span: Span,
    pub traits_len: usize,
    pub traits: DiagSymbolList<String>,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(hir_typeck_suggestion, applicability = "has-placeholders")]
pub(crate) struct LossyProvenanceInt2PtrSuggestion {
    #[suggestion_part(code = "(...).with_addr(")]
    pub lo: Span,
    #[suggestion_part(code = ")")]
    pub hi: Span,
}

#[derive(LintDiagnostic)]
#[diag(hir_typeck_lossy_provenance_ptr2int)]
#[help]
pub(crate) struct LossyProvenancePtr2Int<'tcx> {
    pub expr_ty: Ty<'tcx>,
    pub cast_ty: Ty<'tcx>,
    #[subdiagnostic]
    pub sugg: LossyProvenancePtr2IntSuggestion<'tcx>,
}

#[derive(Subdiagnostic)]
pub(crate) enum LossyProvenancePtr2IntSuggestion<'tcx> {
    #[multipart_suggestion(hir_typeck_suggestion, applicability = "maybe-incorrect")]
    NeedsParensCast {
        #[suggestion_part(code = "(")]
        expr_span: Span,
        #[suggestion_part(code = ").addr() as {cast_ty}")]
        cast_span: Span,
        cast_ty: Ty<'tcx>,
    },
    #[multipart_suggestion(hir_typeck_suggestion, applicability = "maybe-incorrect")]
    NeedsParens {
        #[suggestion_part(code = "(")]
        expr_span: Span,
        #[suggestion_part(code = ").addr()")]
        cast_span: Span,
    },
    #[suggestion(
        hir_typeck_suggestion,
        code = ".addr() as {cast_ty}",
        applicability = "maybe-incorrect"
    )]
    NeedsCast {
        #[primary_span]
        cast_span: Span,
        cast_ty: Ty<'tcx>,
    },
    #[suggestion(hir_typeck_suggestion, code = ".addr()", applicability = "maybe-incorrect")]
    Other {
        #[primary_span]
        cast_span: Span,
    },
}

#[derive(Subdiagnostic)]
pub(crate) enum HelpUseLatestEdition {
    #[help(hir_typeck_help_set_edition_cargo)]
    #[note(hir_typeck_note_edition_guide)]
    Cargo { edition: Edition },
    #[help(hir_typeck_help_set_edition_standalone)]
    #[note(hir_typeck_note_edition_guide)]
    Standalone { edition: Edition },
}

impl HelpUseLatestEdition {
    pub(crate) fn new() -> Self {
        let edition = LATEST_STABLE_EDITION;
        if rustc_session::utils::was_invoked_from_cargo() {
            Self::Cargo { edition }
        } else {
            Self::Standalone { edition }
        }
    }
}

#[derive(Diagnostic)]
#[diag(hir_typeck_no_field_on_type, code = E0609)]
pub(crate) struct NoFieldOnType<'tcx> {
    #[primary_span]
    pub(crate) span: Span,
    pub(crate) ty: Ty<'tcx>,
    pub(crate) field: Ident,
}

#[derive(Diagnostic)]
#[diag(hir_typeck_no_field_on_variant, code = E0609)]
pub(crate) struct NoFieldOnVariant<'tcx> {
    #[primary_span]
    pub(crate) span: Span,
    pub(crate) container: Ty<'tcx>,
    pub(crate) ident: Ident,
    pub(crate) field: Ident,
    #[label(hir_typeck_no_field_on_variant_enum)]
    pub(crate) enum_span: Span,
    #[label(hir_typeck_no_field_on_variant_field)]
    pub(crate) field_span: Span,
}

#[derive(Diagnostic)]
#[diag(hir_typeck_cant_dereference, code = E0614)]
pub(crate) struct CantDereference<'tcx> {
    #[primary_span]
    #[label(hir_typeck_cant_dereference_label)]
    pub(crate) span: Span,
    pub(crate) ty: Ty<'tcx>,
}

#[derive(Diagnostic)]
#[diag(hir_typeck_expected_array_or_slice, code = E0529)]
pub(crate) struct ExpectedArrayOrSlice<'tcx> {
    #[primary_span]
    #[label(hir_typeck_expected_array_or_slice_label)]
    pub(crate) span: Span,
    pub(crate) ty: Ty<'tcx>,
    pub(crate) slice_pat_semantics: bool,
    #[subdiagnostic]
    pub(crate) as_deref: Option<AsDerefSuggestion>,
    #[subdiagnostic]
    pub(crate) slicing: Option<SlicingSuggestion>,
}

#[derive(Subdiagnostic)]
#[suggestion(
    hir_typeck_as_deref_suggestion,
    code = ".as_deref()",
    style = "verbose",
    applicability = "maybe-incorrect"
)]
pub(crate) struct AsDerefSuggestion {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Subdiagnostic)]
#[suggestion(
    hir_typeck_slicing_suggestion,
    code = "[..]",
    style = "verbose",
    applicability = "maybe-incorrect"
)]
pub(crate) struct SlicingSuggestion {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag(hir_typeck_invalid_callee, code = E0618)]
pub(crate) struct InvalidCallee<'tcx> {
    #[primary_span]
    pub span: Span,
    pub ty: Ty<'tcx>,
    pub found: String,
}

#[derive(Diagnostic)]
#[diag(hir_typeck_int_to_fat, code = E0606)]
pub(crate) struct IntToWide<'tcx> {
    #[primary_span]
    #[label(hir_typeck_int_to_fat_label)]
    pub span: Span,
    pub metadata: &'tcx str,
    pub expr_ty: Ty<'tcx>,
    pub cast_ty: Ty<'tcx>,
    #[label(hir_typeck_int_to_fat_label_nightly)]
    pub expr_if_nightly: Option<Span>,
    pub known_wide: bool,
}

#[derive(Subdiagnostic)]
pub(crate) enum OptionResultRefMismatch {
    #[suggestion(
        hir_typeck_option_result_copied,
        code = ".copied()",
        style = "verbose",
        applicability = "machine-applicable"
    )]
    Copied {
        #[primary_span]
        span: Span,
        def_path: String,
    },
    #[suggestion(
        hir_typeck_option_result_cloned,
        code = ".cloned()",
        style = "verbose",
        applicability = "machine-applicable"
    )]
    Cloned {
        #[primary_span]
        span: Span,
        def_path: String,
    },
    // FIXME: #114050
    // #[suggestion(
    //     hir_typeck_option_result_asref,
    //     code = ".as_ref()",
    //     style = "verbose",
    //     applicability = "machine-applicable"
    // )]
    // AsRef {
    //     #[primary_span]
    //     span: Span,
    //     def_path: String,
    //     expected_ty: Ty<'tcx>,
    //     expr_ty: Ty<'tcx>,
    // },
}

pub(crate) struct RemoveSemiForCoerce {
    pub expr: Span,
    pub ret: Span,
    pub semi: Span,
}

impl Subdiagnostic for RemoveSemiForCoerce {
    fn add_to_diag_with<G: EmissionGuarantee, F: SubdiagMessageOp<G>>(
        self,
        diag: &mut Diag<'_, G>,
        _f: &F,
    ) {
        let mut multispan: MultiSpan = self.semi.into();
        multispan.push_span_label(self.expr, fluent::hir_typeck_remove_semi_for_coerce_expr);
        multispan.push_span_label(self.ret, fluent::hir_typeck_remove_semi_for_coerce_ret);
        multispan.push_span_label(self.semi, fluent::hir_typeck_remove_semi_for_coerce_semi);
        diag.span_note(multispan, fluent::hir_typeck_remove_semi_for_coerce);

        diag.tool_only_span_suggestion(
            self.semi,
            fluent::hir_typeck_remove_semi_for_coerce_suggestion,
            "",
            Applicability::MaybeIncorrect,
        );
    }
}

#[derive(Diagnostic)]
#[diag(hir_typeck_const_select_must_be_const)]
#[help]
pub(crate) struct ConstSelectMustBeConst {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(hir_typeck_const_select_must_be_fn)]
#[note]
#[help]
pub(crate) struct ConstSelectMustBeFn<'a> {
    #[primary_span]
    pub span: Span,
    pub ty: Ty<'a>,
}

#[derive(Diagnostic)]
#[diag(hir_typeck_union_pat_multiple_fields)]
pub(crate) struct UnionPatMultipleFields {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(hir_typeck_union_pat_dotdot)]
pub(crate) struct UnionPatDotDot {
    #[primary_span]
    pub span: Span,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    hir_typeck_use_is_empty,
    applicability = "maybe-incorrect",
    style = "verbose"
)]
pub(crate) struct UseIsEmpty<'tcx> {
    #[suggestion_part(code = "!")]
    pub lo: Span,
    #[suggestion_part(code = ".is_empty()")]
    pub hi: Span,
    pub expr_ty: Ty<'tcx>,
}

#[derive(Diagnostic)]
#[diag(hir_typeck_arg_mismatch_indeterminate)]
pub(crate) struct ArgMismatchIndeterminate {
    #[primary_span]
    pub span: Span,
}

#[derive(Subdiagnostic)]
pub(crate) enum SuggestBoxing {
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
    style = "verbose",
    code = "core::ptr::null_mut()"
)]
pub(crate) struct SuggestPtrNullMut {
    #[primary_span]
    pub span: Span,
}

#[derive(LintDiagnostic)]
#[diag(hir_typeck_trivial_cast)]
#[help]
pub(crate) struct TrivialCast<'tcx> {
    pub numeric: bool,
    pub expr_ty: Ty<'tcx>,
    pub cast_ty: Ty<'tcx>,
}

#[derive(Diagnostic)]
#[diag(hir_typeck_no_associated_item, code = E0599)]
pub(crate) struct NoAssociatedItem {
    #[primary_span]
    pub span: Span,
    pub item_kind: &'static str,
    pub item_ident: Ident,
    pub ty_prefix: Cow<'static, str>,
    pub ty_str: String,
    pub trait_missing_method: bool,
}

#[derive(Subdiagnostic)]
#[note(hir_typeck_candidate_trait_note)]
pub(crate) struct CandidateTraitNote {
    #[primary_span]
    pub span: Span,
    pub trait_name: String,
    pub item_name: Ident,
    pub action_or_ty: String,
}

#[derive(Diagnostic)]
#[diag(hir_typeck_cannot_cast_to_bool, code = E0054)]
pub(crate) struct CannotCastToBool<'tcx> {
    #[primary_span]
    pub span: Span,
    pub expr_ty: Ty<'tcx>,
    #[subdiagnostic]
    pub help: CannotCastToBoolHelp,
}

#[derive(Diagnostic)]
#[diag(hir_typeck_cast_enum_drop)]
pub(crate) struct CastEnumDrop<'tcx> {
    #[primary_span]
    pub span: Span,
    pub expr_ty: Ty<'tcx>,
    pub cast_ty: Ty<'tcx>,
}

#[derive(Diagnostic)]
#[diag(hir_typeck_cast_unknown_pointer, code = E0641)]
pub(crate) struct CastUnknownPointer {
    #[primary_span]
    pub span: Span,
    pub to: bool,
    #[subdiagnostic]
    pub sub: CastUnknownPointerSub,
}

pub(crate) enum CastUnknownPointerSub {
    To(Span),
    From(Span),
}

impl rustc_errors::Subdiagnostic for CastUnknownPointerSub {
    fn add_to_diag_with<G: EmissionGuarantee, F: SubdiagMessageOp<G>>(
        self,
        diag: &mut Diag<'_, G>,
        f: &F,
    ) {
        match self {
            CastUnknownPointerSub::To(span) => {
                let msg = f(diag, crate::fluent_generated::hir_typeck_label_to);
                diag.span_label(span, msg);
                let msg = f(diag, crate::fluent_generated::hir_typeck_note);
                diag.note(msg);
            }
            CastUnknownPointerSub::From(span) => {
                let msg = f(diag, crate::fluent_generated::hir_typeck_label_from);
                diag.span_label(span, msg);
            }
        }
    }
}

#[derive(Subdiagnostic)]
pub(crate) enum CannotCastToBoolHelp {
    #[suggestion(
        hir_typeck_suggestion,
        applicability = "machine-applicable",
        code = " != 0",
        style = "verbose"
    )]
    Numeric(#[primary_span] Span),
    #[label(hir_typeck_label)]
    Unsupported(#[primary_span] Span),
}

#[derive(Diagnostic)]
#[diag(hir_typeck_ctor_is_private, code = E0603)]
pub(crate) struct CtorIsPrivate {
    #[primary_span]
    pub span: Span,
    pub def: String,
}

#[derive(Subdiagnostic)]
#[note(hir_typeck_deref_is_empty)]
pub(crate) struct DerefImplsIsEmpty<'tcx> {
    #[primary_span]
    pub span: Span,
    pub deref_ty: Ty<'tcx>,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    hir_typeck_convert_using_method,
    applicability = "machine-applicable",
    style = "verbose"
)]
pub(crate) struct SuggestConvertViaMethod<'tcx> {
    #[suggestion_part(code = "{sugg}")]
    pub span: Span,
    #[suggestion_part(code = "")]
    pub borrow_removal_span: Option<Span>,
    pub sugg: String,
    pub expected: Ty<'tcx>,
    pub found: Ty<'tcx>,
}

#[derive(Subdiagnostic)]
#[note(hir_typeck_note_caller_chooses_ty_for_ty_param)]
pub(crate) struct NoteCallerChoosesTyForTyParam<'tcx> {
    pub ty_param_name: Symbol,
    pub found_ty: Ty<'tcx>,
}

#[derive(Subdiagnostic)]
pub(crate) enum SuggestBoxingForReturnImplTrait {
    #[multipart_suggestion(hir_typeck_rpit_change_return_type, applicability = "maybe-incorrect")]
    ChangeReturnType {
        #[suggestion_part(code = "Box<dyn")]
        start_sp: Span,
        #[suggestion_part(code = ">")]
        end_sp: Span,
    },
    #[multipart_suggestion(hir_typeck_rpit_box_return_expr, applicability = "maybe-incorrect")]
    BoxReturnExpr {
        #[suggestion_part(code = "Box::new(")]
        starts: Vec<Span>,
        #[suggestion_part(code = ")")]
        ends: Vec<Span>,
    },
}

#[derive(Diagnostic)]
#[diag(hir_typeck_self_ctor_from_outer_item, code = E0401)]
pub(crate) struct SelfCtorFromOuterItem {
    #[primary_span]
    pub span: Span,
    #[label]
    pub impl_span: Span,
    #[subdiagnostic]
    pub sugg: Option<ReplaceWithName>,
}

#[derive(LintDiagnostic)]
#[diag(hir_typeck_self_ctor_from_outer_item)]
pub(crate) struct SelfCtorFromOuterItemLint {
    #[label]
    pub impl_span: Span,
    #[subdiagnostic]
    pub sugg: Option<ReplaceWithName>,
}

#[derive(Subdiagnostic)]
#[suggestion(hir_typeck_suggestion, code = "{name}", applicability = "machine-applicable")]
pub(crate) struct ReplaceWithName {
    #[primary_span]
    pub span: Span,
    pub name: String,
}

#[derive(Diagnostic)]
#[diag(hir_typeck_cast_thin_pointer_to_wide_pointer, code = E0607)]
pub(crate) struct CastThinPointerToWidePointer<'tcx> {
    #[primary_span]
    pub span: Span,
    pub expr_ty: Ty<'tcx>,
    pub cast_ty: Ty<'tcx>,
    #[note(hir_typeck_teach_help)]
    pub(crate) teach: bool,
}

#[derive(Diagnostic)]
#[diag(hir_typeck_pass_to_variadic_function, code = E0617)]
pub(crate) struct PassToVariadicFunction<'a, 'tcx> {
    #[primary_span]
    pub span: Span,
    pub ty: Ty<'tcx>,
    pub cast_ty: &'a str,
    #[suggestion(code = " as {cast_ty}", applicability = "machine-applicable", style = "verbose")]
    pub sugg_span: Span,
    #[note(hir_typeck_teach_help)]
    pub(crate) teach: bool,
}

#[derive(Diagnostic)]
#[diag(hir_typeck_fn_item_to_variadic_function, code = E0617)]
#[help]
#[note]
pub(crate) struct PassFnItemToVariadicFunction {
    #[primary_span]
    pub span: Span,
    #[suggestion(code = " as {replace}", applicability = "machine-applicable", style = "verbose")]
    pub sugg_span: Span,
    pub replace: String,
}

#[derive(Subdiagnostic)]
#[suggestion(
    hir_typeck_replace_comma_with_semicolon,
    applicability = "machine-applicable",
    style = "verbose",
    code = "; "
)]
pub(crate) struct ReplaceCommaWithSemicolon {
    #[primary_span]
    pub comma_span: Span,
    pub descr: &'static str,
}

#[derive(LintDiagnostic)]
#[diag(hir_typeck_supertrait_item_shadowing)]
pub(crate) struct SupertraitItemShadowing {
    pub item: Symbol,
    pub subtrait: Symbol,
    #[subdiagnostic]
    pub shadower: SupertraitItemShadower,
    #[subdiagnostic]
    pub shadowee: SupertraitItemShadowee,
}

#[derive(Subdiagnostic)]
#[note(hir_typeck_supertrait_item_shadower)]
pub(crate) struct SupertraitItemShadower {
    pub subtrait: Symbol,
    #[primary_span]
    pub span: Span,
}

#[derive(Subdiagnostic)]
pub(crate) enum SupertraitItemShadowee {
    #[note(hir_typeck_supertrait_item_shadowee)]
    Labeled {
        #[primary_span]
        span: Span,
        supertrait: Symbol,
    },
    #[note(hir_typeck_supertrait_item_multiple_shadowee)]
    Several {
        #[primary_span]
        spans: MultiSpan,
        traits: DiagSymbolList,
    },
}
