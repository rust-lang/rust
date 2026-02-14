//! Errors emitted by `rustc_hir_typeck`.

use std::borrow::Cow;

use rustc_abi::ExternAbi;
use rustc_ast::{AssignOpKind, Label};
use rustc_errors::codes::*;
use rustc_errors::{
    Applicability, Diag, DiagArgValue, DiagCtxtHandle, DiagSymbolList, Diagnostic,
    EmissionGuarantee, IntoDiagArg, Level, MultiSpan, Subdiagnostic, msg,
};
use rustc_hir as hir;
use rustc_hir::ExprKind;
use rustc_macros::{Diagnostic, LintDiagnostic, Subdiagnostic};
use rustc_middle::ty::{self, Ty};
use rustc_span::edition::{Edition, LATEST_STABLE_EDITION};
use rustc_span::source_map::Spanned;
use rustc_span::{Ident, Span, Symbol};

use crate::FnCtxt;

#[derive(Diagnostic)]
#[diag("base expression required after `..`", code = E0797)]
pub(crate) struct BaseExpressionDoubleDot {
    #[primary_span]
    pub span: Span,
    #[suggestion(
        "add `#![feature(default_field_values)]` to the crate attributes to enable default values on `struct` fields",
        code = "#![feature(default_field_values)]\n",
        applicability = "machine-applicable",
        style = "verbose"
    )]
    pub default_field_values_suggestion: Option<Span>,
    #[subdiagnostic]
    pub add_expr: Option<BaseExpressionDoubleDotAddExpr>,
    #[subdiagnostic]
    pub remove_dots: Option<BaseExpressionDoubleDotRemove>,
}

#[derive(Subdiagnostic)]
#[suggestion(
    "remove the `..` as all the fields are already present",
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
    "add a base expression here",
    code = "/* expr */",
    applicability = "has-placeholders",
    style = "verbose"
)]
pub(crate) struct BaseExpressionDoubleDotAddExpr {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("field `{$ident}` specified more than once", code = E0062)]
pub(crate) struct FieldMultiplySpecifiedInInitializer {
    #[primary_span]
    #[label("used more than once")]
    pub span: Span,
    #[label("first use of `{$ident}`")]
    pub prev_span: Span,
    pub ident: Ident,
}

#[derive(Diagnostic)]
#[diag("{$statement_kind} statement outside of function body", code = E0572)]
pub(crate) struct ReturnStmtOutsideOfFnBody {
    #[primary_span]
    pub span: Span,
    #[label("the {$statement_kind} is part of this body...")]
    pub encl_body_span: Option<Span>,
    #[label("...not the enclosing function body")]
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
#[diag("functions with the \"rust-call\" ABI must take a single non-self tuple argument")]
pub(crate) struct RustCallIncorrectArgs {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("yield expression outside of coroutine literal", code = E0627)]
pub(crate) struct YieldExprOutsideOfCoroutine {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("cannot create non-exhaustive {$what} using struct expression", code = E0639)]
pub(crate) struct StructExprNonExhaustive {
    #[primary_span]
    pub span: Span,
    pub what: &'static str,
}

#[derive(Diagnostic)]
#[diag("functional record update syntax requires a struct", code = E0436)]
pub(crate) struct FunctionalRecordUpdateOnNonStruct {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("cannot take address of a temporary", code = E0745)]
pub(crate) struct AddressOfTemporaryTaken {
    #[primary_span]
    #[label("temporary value")]
    pub span: Span,
}

#[derive(Subdiagnostic)]
pub(crate) enum AddReturnTypeSuggestion {
    #[suggestion(
        "try adding a return type",
        code = " -> {found}",
        applicability = "machine-applicable"
    )]
    Add {
        #[primary_span]
        span: Span,
        found: String,
    },
    #[suggestion(
        "a return type might be missing here",
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
    #[label("expected `()` because of default return type")]
    Unit {
        #[primary_span]
        span: Span,
    },
    #[label("expected `{$expected}` because of return type")]
    Other {
        #[primary_span]
        span: Span,
        expected: Ty<'tcx>,
    },
}

#[derive(Diagnostic)]
#[diag("explicit use of destructor method", code = E0040)]
pub(crate) struct ExplicitDestructorCall {
    #[primary_span]
    #[label("explicit destructor calls not allowed")]
    pub span: Span,
    #[subdiagnostic]
    pub sugg: ExplicitDestructorCallSugg,
}

#[derive(Subdiagnostic)]
pub(crate) enum ExplicitDestructorCallSugg {
    #[suggestion(
        "consider using `drop` function",
        code = "drop",
        applicability = "maybe-incorrect"
    )]
    Empty(#[primary_span] Span),
    #[multipart_suggestion("consider using `drop` function", style = "short")]
    Snippet {
        #[suggestion_part(code = "drop(")]
        lo: Span,
        #[suggestion_part(code = ")")]
        hi: Span,
    },
}

#[derive(Diagnostic)]
#[diag("can't call method `{$method_name}` on type `{$ty}`", code = E0689)]
pub(crate) struct MissingParenthesesInRange<'tcx> {
    #[primary_span]
    #[label("can't call method `{$method_name}` on type `{$ty}`")]
    pub span: Span,
    pub ty: Ty<'tcx>,
    pub method_name: String,
    #[subdiagnostic]
    pub add_missing_parentheses: Option<AddMissingParenthesesInRange>,
}

#[derive(LintDiagnostic)]
pub(crate) enum NeverTypeFallbackFlowingIntoUnsafe {
    #[help("specify the type explicitly")]
    #[diag("never type fallback affects this call to an `unsafe` function")]
    Call {
        #[subdiagnostic]
        sugg: SuggestAnnotations,
    },
    #[help("specify the type explicitly")]
    #[diag("never type fallback affects this call to an `unsafe` method")]
    Method {
        #[subdiagnostic]
        sugg: SuggestAnnotations,
    },
    #[help("specify the type explicitly")]
    #[diag("never type fallback affects this `unsafe` function")]
    Path {
        #[subdiagnostic]
        sugg: SuggestAnnotations,
    },
    #[help("specify the type explicitly")]
    #[diag("never type fallback affects this union access")]
    UnionField {
        #[subdiagnostic]
        sugg: SuggestAnnotations,
    },
    #[help("specify the type explicitly")]
    #[diag("never type fallback affects this raw pointer dereference")]
    Deref {
        #[subdiagnostic]
        sugg: SuggestAnnotations,
    },
}

#[derive(LintDiagnostic)]
#[help("specify the types explicitly")]
#[diag("this function depends on never type fallback being `()`")]
pub(crate) struct DependencyOnUnitNeverTypeFallback<'tcx> {
    #[note("in edition 2024, the requirement `{$obligation}` will fail")]
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
    fn add_to_diag<G: EmissionGuarantee>(self, diag: &mut Diag<'_, G>) {
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
    "you must surround the range in parentheses to call its `{$func_name}` function",
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
    fn add_to_diag<G: EmissionGuarantee>(self, diag: &mut Diag<'_, G>) {
        diag.arg("expr", self.expr.as_deref().unwrap_or("NONE"));

        // Only explain that `a ..b` is a range if it's split up
        if self.expr_span.between(self.fru_span).is_empty() {
            diag.span_note(
                self.expr_span.to(self.fru_span),
                msg!("this expression may have been misinterpreted as a `..` range expression"),
            );
        } else {
            let mut multispan: MultiSpan = vec![self.expr_span, self.fru_span].into();
            multispan.push_span_label(
                self.expr_span,
                msg!("this expression does not end in a comma..."),
            );
            multispan.push_span_label(self.fru_span, msg!("... so this is interpreted as a `..` range expression, instead of functional record update syntax"));
            diag.span_note(
                multispan,
                msg!("this expression may have been misinterpreted as a `..` range expression"),
            );
        }

        diag.span_suggestion(
            self.expr_span.shrink_to_hi(),
            msg!(
                "to set the remaining fields{$expr ->
                    [NONE]{\"\"}
                    *[other] {\" \"}from `{$expr}`
                }, separate the last named field with a comma"
            ),
            ", ",
            Applicability::MaybeIncorrect,
        );
    }
}

#[derive(LintDiagnostic)]
#[diag("strict provenance disallows casting integer `{$expr_ty}` to pointer `{$cast_ty}`")]
#[help(
    "if you can't comply with strict provenance and don't have a pointer with the correct provenance you can use `std::ptr::with_exposed_provenance()` instead"
)]
pub(crate) struct LossyProvenanceInt2Ptr<'tcx> {
    pub expr_ty: Ty<'tcx>,
    pub cast_ty: Ty<'tcx>,
    #[subdiagnostic]
    pub sugg: LossyProvenanceInt2PtrSuggestion,
}

#[derive(Diagnostic)]
#[diag("cannot add {$traits_len ->
    [1] auto trait {$traits}
    *[other] auto traits {$traits}
} to dyn bound via pointer cast", code = E0804)]
#[note("this could allow UB elsewhere")]
#[help("use `transmute` if you're sure this is sound")]
pub(crate) struct PtrCastAddAutoToObject {
    #[primary_span]
    #[label("unsupported cast")]
    pub span: Span,
    pub traits_len: usize,
    pub traits: DiagSymbolList<String>,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    "use `.with_addr()` to adjust a valid pointer in the same allocation, to this address",
    applicability = "has-placeholders"
)]
pub(crate) struct LossyProvenanceInt2PtrSuggestion {
    #[suggestion_part(code = "(...).with_addr(")]
    pub lo: Span,
    #[suggestion_part(code = ")")]
    pub hi: Span,
}

#[derive(LintDiagnostic)]
#[diag(
    "under strict provenance it is considered bad style to cast pointer `{$expr_ty}` to integer `{$cast_ty}`"
)]
#[help(
    "if you can't comply with strict provenance and need to expose the pointer provenance you can use `.expose_provenance()` instead"
)]
pub(crate) struct LossyProvenancePtr2Int<'tcx> {
    pub expr_ty: Ty<'tcx>,
    pub cast_ty: Ty<'tcx>,
    #[subdiagnostic]
    pub sugg: LossyProvenancePtr2IntSuggestion<'tcx>,
}

#[derive(Subdiagnostic)]
pub(crate) enum LossyProvenancePtr2IntSuggestion<'tcx> {
    #[multipart_suggestion(
        "use `.addr()` to obtain the address of a pointer",
        applicability = "maybe-incorrect"
    )]
    NeedsParensCast {
        #[suggestion_part(code = "(")]
        expr_span: Span,
        #[suggestion_part(code = ").addr() as {cast_ty}")]
        cast_span: Span,
        cast_ty: Ty<'tcx>,
    },
    #[multipart_suggestion(
        "use `.addr()` to obtain the address of a pointer",
        applicability = "maybe-incorrect"
    )]
    NeedsParens {
        #[suggestion_part(code = "(")]
        expr_span: Span,
        #[suggestion_part(code = ").addr()")]
        cast_span: Span,
    },
    #[suggestion(
        "use `.addr()` to obtain the address of a pointer",
        code = ".addr() as {cast_ty}",
        applicability = "maybe-incorrect"
    )]
    NeedsCast {
        #[primary_span]
        cast_span: Span,
        cast_ty: Ty<'tcx>,
    },
    #[suggestion(
        "use `.addr()` to obtain the address of a pointer",
        code = ".addr()",
        applicability = "maybe-incorrect"
    )]
    Other {
        #[primary_span]
        cast_span: Span,
    },
}

#[derive(Subdiagnostic)]
pub(crate) enum HelpUseLatestEdition {
    #[help("set `edition = \"{$edition}\"` in `Cargo.toml`")]
    #[note("for more on editions, read https://doc.rust-lang.org/edition-guide")]
    Cargo { edition: Edition },
    #[help("pass `--edition {$edition}` to `rustc`")]
    #[note("for more on editions, read https://doc.rust-lang.org/edition-guide")]
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
#[diag("no field `{$field}` on type `{$ty}`", code = E0609)]
pub(crate) struct NoFieldOnType<'tcx> {
    #[primary_span]
    pub(crate) span: Span,
    pub(crate) ty: Ty<'tcx>,
    pub(crate) field: Ident,
}

#[derive(Diagnostic)]
#[diag("no field named `{$field}` on enum variant `{$container}::{$ident}`", code = E0609)]
pub(crate) struct NoFieldOnVariant<'tcx> {
    #[primary_span]
    pub(crate) span: Span,
    pub(crate) container: Ty<'tcx>,
    pub(crate) ident: Ident,
    pub(crate) field: Ident,
    #[label("this enum variant...")]
    pub(crate) enum_span: Span,
    #[label("...does not have this field")]
    pub(crate) field_span: Span,
}

#[derive(Diagnostic)]
#[diag("type `{$ty}` cannot be dereferenced", code = E0614)]
pub(crate) struct CantDereference<'tcx> {
    #[primary_span]
    #[label("can't be dereferenced")]
    pub(crate) span: Span,
    pub(crate) ty: Ty<'tcx>,
}

#[derive(Diagnostic)]
#[diag("expected an array or slice, found `{$ty}`", code = E0529)]
pub(crate) struct ExpectedArrayOrSlice<'tcx> {
    #[primary_span]
    #[label("pattern cannot match with input type `{$ty}`")]
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
    "consider using `as_deref` here",
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
    "consider slicing here",
    code = "[..]",
    style = "verbose",
    applicability = "maybe-incorrect"
)]
pub(crate) struct SlicingSuggestion {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag("expected function, found {$found}", code = E0618)]
pub(crate) struct InvalidCallee<'tcx> {
    #[primary_span]
    pub span: Span,
    pub ty: Ty<'tcx>,
    pub found: String,
}

#[derive(Diagnostic)]
#[diag("cannot cast `{$expr_ty}` to a pointer that {$known_wide ->
    [true] is
    *[false] may be
} wide", code = E0606)]
pub(crate) struct IntToWide<'tcx> {
    #[primary_span]
    #[label("creating a `{$cast_ty}` requires both an address and {$metadata}")]
    pub span: Span,
    pub metadata: &'tcx str,
    pub expr_ty: Ty<'tcx>,
    pub cast_ty: Ty<'tcx>,
    #[label(
        "consider casting this expression to `*const ()`, then using `core::ptr::from_raw_parts`"
    )]
    pub expr_if_nightly: Option<Span>,
    pub known_wide: bool,
    #[subdiagnostic]
    pub param_note: Option<IntToWideParamNote>,
}

#[derive(Subdiagnostic)]
#[note("the type parameter `{$param}` is not known to be `Sized`, so this pointer may be wide")]
pub(crate) struct IntToWideParamNote {
    pub param: Symbol,
}

#[derive(Subdiagnostic)]
pub(crate) enum OptionResultRefMismatch {
    #[suggestion(
        "use `{$def_path}::copied` to copy the value inside the `{$def_path}`",
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
        "use `{$def_path}::cloned` to clone the value inside the `{$def_path}`",
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
    //     "use `{$def_path}::as_ref` to convert `{$expected_ty}` to `{$expr_ty}`",
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
    fn add_to_diag<G: EmissionGuarantee>(self, diag: &mut Diag<'_, G>) {
        let mut multispan: MultiSpan = self.semi.into();
        multispan.push_span_label(
            self.expr,
            msg!("this could be implicitly returned but it is a statement, not a tail expression"),
        );
        multispan
            .push_span_label(self.ret, msg!("the `match` arms can conform to this return type"));
        multispan.push_span_label(
            self.semi,
            msg!("the `match` is a statement because of this semicolon, consider removing it"),
        );
        diag.span_note(multispan, msg!("you might have meant to return the `match` expression"));

        diag.tool_only_span_suggestion(
            self.semi,
            msg!("remove this semicolon"),
            "",
            Applicability::MaybeIncorrect,
        );
    }
}

#[derive(Diagnostic)]
#[diag("union patterns should have exactly one field")]
pub(crate) struct UnionPatMultipleFields {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("`..` cannot be used in union patterns")]
pub(crate) struct UnionPatDotDot {
    #[primary_span]
    pub span: Span,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    "consider using the `is_empty` method on `{$expr_ty}` to determine if it contains anything",
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
#[diag("argument type mismatch was detected, but rustc had trouble determining where")]
pub(crate) struct ArgMismatchIndeterminate {
    #[primary_span]
    pub span: Span,
}

#[derive(Subdiagnostic)]
pub(crate) enum SuggestBoxing {
    #[note(
        "for more on the distinction between the stack and the heap, read https://doc.rust-lang.org/book/ch15-01-box.html, https://doc.rust-lang.org/rust-by-example/std/box.html, and https://doc.rust-lang.org/std/boxed/index.html"
    )]
    #[multipart_suggestion(
        "store this in the heap by calling `Box::new`",
        applicability = "machine-applicable"
    )]
    Unit {
        #[suggestion_part(code = "Box::new(())")]
        start: Span,
        #[suggestion_part(code = "")]
        end: Span,
    },
    #[note(
        "for more on the distinction between the stack and the heap, read https://doc.rust-lang.org/book/ch15-01-box.html, https://doc.rust-lang.org/rust-by-example/std/box.html, and https://doc.rust-lang.org/std/boxed/index.html"
    )]
    AsyncBody,
    #[note(
        "for more on the distinction between the stack and the heap, read https://doc.rust-lang.org/book/ch15-01-box.html, https://doc.rust-lang.org/rust-by-example/std/box.html, and https://doc.rust-lang.org/std/boxed/index.html"
    )]
    #[multipart_suggestion(
        "store this in the heap by calling `Box::new`",
        applicability = "machine-applicable"
    )]
    ExprFieldShorthand {
        #[suggestion_part(code = "{ident}: Box::new(")]
        start: Span,
        #[suggestion_part(code = ")")]
        end: Span,
        ident: Ident,
    },
    #[note(
        "for more on the distinction between the stack and the heap, read https://doc.rust-lang.org/book/ch15-01-box.html, https://doc.rust-lang.org/rust-by-example/std/box.html, and https://doc.rust-lang.org/std/boxed/index.html"
    )]
    #[multipart_suggestion(
        "store this in the heap by calling `Box::new`",
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
    "consider using `core::ptr::null_mut` instead",
    applicability = "maybe-incorrect",
    style = "verbose",
    code = "core::ptr::null_mut()"
)]
pub(crate) struct SuggestPtrNullMut {
    #[primary_span]
    pub span: Span,
}

#[derive(LintDiagnostic)]
#[diag(
    "trivial {$numeric ->
        [true] numeric cast
        *[false] cast
    }: `{$expr_ty}` as `{$cast_ty}`"
)]
#[help("cast can be replaced by coercion; this might require a temporary variable")]
pub(crate) struct TrivialCast<'tcx> {
    pub numeric: bool,
    pub expr_ty: Ty<'tcx>,
    pub cast_ty: Ty<'tcx>,
}

pub(crate) struct BreakNonLoop<'a> {
    pub span: Span,
    pub head: Option<Span>,
    pub kind: &'a str,
    pub suggestion: String,
    pub loop_label: Option<Label>,
    pub break_label: Option<Label>,
    pub break_expr_kind: &'a ExprKind<'a>,
    pub break_expr_span: Span,
}

impl<'a, G: EmissionGuarantee> Diagnostic<'_, G> for BreakNonLoop<'a> {
    #[track_caller]
    fn into_diag(self, dcx: DiagCtxtHandle<'_>, level: Level) -> Diag<'_, G> {
        let mut diag = Diag::new(dcx, level, msg!("`break` with value from a `{$kind}` loop"));
        diag.span(self.span);
        diag.code(E0571);
        diag.arg("kind", self.kind);
        diag.span_label(
            self.span,
            msg!("can only break with a value inside `loop` or breakable block"),
        );
        if let Some(head) = self.head {
            diag.span_label(head, msg!("you can't `break` with a value in a `{$kind}` loop"));
        }
        diag.span_suggestion(
            self.span,
            msg!("use `break` on its own without a value inside this `{$kind}` loop"),
            self.suggestion,
            Applicability::MaybeIncorrect,
        );
        if let (Some(label), None) = (self.loop_label, self.break_label) {
            match self.break_expr_kind {
                ExprKind::Path(hir::QPath::Resolved(
                    None,
                    hir::Path { segments: [segment], res: hir::def::Res::Err, .. },
                )) if label.ident.to_string() == format!("'{}", segment.ident) => {
                    // This error is redundant, we will have already emitted a
                    // suggestion to use the label when `segment` wasn't found
                    // (hence the `Res::Err` check).
                    diag.downgrade_to_delayed_bug();
                }
                _ => {
                    diag.span_suggestion(
                        self.break_expr_span,
                        msg!("alternatively, you might have meant to use the available loop label"),
                        label.ident,
                        Applicability::MaybeIncorrect,
                    );
                }
            }
        }
        diag
    }
}

#[derive(Diagnostic)]
#[diag("`continue` pointing to a labeled block", code = E0696)]
pub(crate) struct ContinueLabeledBlock {
    #[primary_span]
    #[label("labeled blocks cannot be `continue`'d")]
    pub span: Span,
    #[label("labeled block the `continue` points to")]
    pub block_span: Span,
}

#[derive(Diagnostic)]
#[diag("`{$name}` inside of a closure", code = E0267)]
pub(crate) struct BreakInsideClosure<'a> {
    #[primary_span]
    #[label("cannot `{$name}` inside of a closure")]
    pub span: Span,
    #[label("enclosing closure")]
    pub closure_span: Span,
    pub name: &'a str,
}

#[derive(Diagnostic)]
#[diag("`{$name}` inside `{$kind}` {$source}", code = E0267)]
pub(crate) struct BreakInsideCoroutine<'a> {
    #[primary_span]
    #[label("cannot `{$name}` inside `{$kind}` {$source}")]
    pub span: Span,
    #[label("enclosing `{$kind}` {$source}")]
    pub coroutine_span: Span,
    pub name: &'a str,
    pub kind: &'a str,
    pub source: &'a str,
}

#[derive(Diagnostic)]
#[diag("`{$name}` outside of a loop{$is_break ->
        [true] {\" or labeled block\"}
        *[false] {\"\"}
    }", code = E0268)]
pub(crate) struct OutsideLoop<'a> {
    #[primary_span]
    #[label(
        "cannot `{$name}` outside of a loop{$is_break ->
        [true] {\" or labeled block\"}
        *[false] {\"\"}
    }"
    )]
    pub spans: Vec<Span>,
    pub name: &'a str,
    pub is_break: bool,
    #[subdiagnostic]
    pub suggestion: Option<OutsideLoopSuggestion>,
}
#[derive(Subdiagnostic)]
#[multipart_suggestion(
    "consider labeling this block to be able to break within it",
    applicability = "maybe-incorrect"
)]
pub(crate) struct OutsideLoopSuggestion {
    #[suggestion_part(code = "'block: ")]
    pub block_span: Span,
    #[suggestion_part(code = " 'block")]
    pub break_spans: Vec<Span>,
}

#[derive(Diagnostic)]
#[diag("unlabeled `{$cf_type}` inside of a labeled block", code = E0695)]
pub(crate) struct UnlabeledInLabeledBlock<'a> {
    #[primary_span]
    #[label(
        "`{$cf_type}` statements that would diverge to or through a labeled block need to bear a label"
    )]
    pub span: Span,
    pub cf_type: &'a str,
}

#[derive(Diagnostic)]
#[diag("`break` or `continue` with no label in the condition of a `while` loop", code = E0590)]
pub(crate) struct UnlabeledCfInWhileCondition<'a> {
    #[primary_span]
    #[label("unlabeled `{$cf_type}` in the condition of a `while` loop")]
    pub span: Span,
    pub cf_type: &'a str,
}

#[derive(Diagnostic)]
#[diag("no {$item_kind} named `{$item_ident}` found for {$ty_prefix} `{$ty}`{$trait_missing_method ->
    [true] {\"\"}
    *[other] {\" \"}in the current scope
}", code = E0599)]
pub(crate) struct NoAssociatedItem<'tcx> {
    #[primary_span]
    pub span: Span,
    pub item_kind: &'static str,
    pub item_ident: Ident,
    pub ty_prefix: Cow<'static, str>,
    pub ty: Ty<'tcx>,
    pub trait_missing_method: bool,
}

#[derive(Subdiagnostic)]
#[note(
    "`{$trait_name}` defines an item `{$item_name}`{$action_or_ty ->
        [NONE] {\"\"}
        [implement] , perhaps you need to implement it
        *[other] , perhaps you need to restrict type parameter `{$action_or_ty}` with it
    }"
)]
pub(crate) struct CandidateTraitNote {
    #[primary_span]
    pub span: Span,
    pub trait_name: String,
    pub item_name: Ident,
    pub action_or_ty: String,
}

#[derive(Diagnostic)]
#[diag("cannot cast `{$expr_ty}` as `bool`", code = E0054)]
pub(crate) struct CannotCastToBool<'tcx> {
    #[primary_span]
    pub span: Span,
    pub expr_ty: Ty<'tcx>,
    #[subdiagnostic]
    pub help: CannotCastToBoolHelp,
}

#[derive(Diagnostic)]
#[diag("cannot cast enum `{$expr_ty}` into integer `{$cast_ty}` because it implements `Drop`")]
pub(crate) struct CastEnumDrop<'tcx> {
    #[primary_span]
    pub span: Span,
    pub expr_ty: Ty<'tcx>,
    pub cast_ty: Ty<'tcx>,
}

#[derive(Diagnostic)]
#[diag("cannot cast {$to ->
    [true] to
    *[false] from
} a pointer of an unknown kind", code = E0641)]
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
    fn add_to_diag<G: EmissionGuarantee>(self, diag: &mut Diag<'_, G>) {
        match self {
            CastUnknownPointerSub::To(span) => {
                let msg = diag.eagerly_translate(msg!("needs more type information"));
                diag.span_label(span, msg);
                let msg = diag.eagerly_translate(msg!("the type information given here is insufficient to check whether the pointer cast is valid"));
                diag.note(msg);
            }
            CastUnknownPointerSub::From(span) => {
                let msg = diag.eagerly_translate(msg!("the type information given here is insufficient to check whether the pointer cast is valid"));
                diag.span_label(span, msg);
            }
        }
    }
}

#[derive(Subdiagnostic)]
pub(crate) enum CannotCastToBoolHelp {
    #[suggestion(
        "compare with zero instead",
        applicability = "machine-applicable",
        code = " != 0",
        style = "verbose"
    )]
    Numeric(#[primary_span] Span),
    #[label("unsupported cast")]
    Unsupported(#[primary_span] Span),
}

#[derive(Diagnostic)]
#[diag("tuple struct constructor `{$def}` is private", code = E0603)]
pub(crate) struct CtorIsPrivate {
    #[primary_span]
    pub span: Span,
    pub def: String,
}

#[derive(Subdiagnostic)]
#[note("this expression `Deref`s to `{$deref_ty}` which implements `is_empty`")]
pub(crate) struct DerefImplsIsEmpty<'tcx> {
    #[primary_span]
    pub span: Span,
    pub deref_ty: Ty<'tcx>,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    "try using `{$sugg}` to convert `{$found}` to `{$expected}`",
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
#[note(
    "the caller chooses a type for `{$ty_param_name}` which can be different from `{$found_ty}`"
)]
pub(crate) struct NoteCallerChoosesTyForTyParam<'tcx> {
    pub ty_param_name: Symbol,
    pub found_ty: Ty<'tcx>,
}

#[derive(Subdiagnostic)]
pub(crate) enum SuggestBoxingForReturnImplTrait {
    #[multipart_suggestion(
        "you could change the return type to be a boxed trait object",
        applicability = "maybe-incorrect"
    )]
    ChangeReturnType {
        #[suggestion_part(code = "Box<dyn")]
        start_sp: Span,
        #[suggestion_part(code = ">")]
        end_sp: Span,
    },
    #[multipart_suggestion(
        "if you change the return type to expect trait objects, box the returned expressions",
        applicability = "maybe-incorrect"
    )]
    BoxReturnExpr {
        #[suggestion_part(code = "Box::new(")]
        starts: Vec<Span>,
        #[suggestion_part(code = ")")]
        ends: Vec<Span>,
    },
}

#[derive(Diagnostic)]
#[diag("can't reference `Self` constructor from outer item", code = E0401)]
pub(crate) struct SelfCtorFromOuterItem {
    #[primary_span]
    pub span: Span,
    #[label(
        "the inner item doesn't inherit generics from this impl, so `Self` is invalid to reference"
    )]
    pub impl_span: Span,
    #[subdiagnostic]
    pub sugg: Option<ReplaceWithName>,
    #[subdiagnostic]
    pub item: Option<InnerItem>,
}

#[derive(Subdiagnostic)]
#[label("`Self` used in this inner item")]
pub(crate) struct InnerItem {
    #[primary_span]
    pub span: Span,
}

#[derive(LintDiagnostic)]
#[diag("can't reference `Self` constructor from outer item")]
pub(crate) struct SelfCtorFromOuterItemLint {
    #[label(
        "the inner item doesn't inherit generics from this impl, so `Self` is invalid to reference"
    )]
    pub impl_span: Span,
    #[subdiagnostic]
    pub sugg: Option<ReplaceWithName>,
    #[subdiagnostic]
    pub item: Option<InnerItem>,
}

#[derive(Subdiagnostic)]
#[suggestion(
    "replace `Self` with the actual type",
    code = "{name}",
    applicability = "machine-applicable"
)]
pub(crate) struct ReplaceWithName {
    #[primary_span]
    pub span: Span,
    pub name: String,
}

#[derive(Diagnostic)]
#[diag("cannot cast thin pointer `{$expr_ty}` to wide pointer `{$cast_ty}`", code = E0607)]
pub(crate) struct CastThinPointerToWidePointer<'tcx> {
    #[primary_span]
    pub span: Span,
    pub expr_ty: Ty<'tcx>,
    pub cast_ty: Ty<'tcx>,
    #[note(
        "thin pointers are \"simple\" pointers: they are purely a reference to a
        memory address.

        Wide pointers are pointers referencing \"Dynamically Sized Types\" (also
        called DST). DST don't have a statically known size, therefore they can
        only exist behind some kind of pointers that contain additional
        information. Slices and trait objects are DSTs. In the case of slices,
        the additional information the wide pointer holds is their size.

        To fix this error, don't try to cast directly between thin and wide
        pointers.

        For more information about casts, take a look at The Book:
        https://doc.rust-lang.org/reference/expressions/operator-expr.html#type-cast-expressions"
    )]
    pub(crate) teach: bool,
}

#[derive(Diagnostic)]
#[diag("can't pass `{$ty}` to variadic function", code = E0617)]
pub(crate) struct PassToVariadicFunction<'a, 'tcx> {
    #[primary_span]
    pub span: Span,
    pub ty: Ty<'tcx>,
    pub cast_ty: &'a str,
    #[suggestion(
        "cast the value to `{$cast_ty}`",
        code = " as {cast_ty}",
        applicability = "machine-applicable",
        style = "verbose"
    )]
    pub sugg_span: Span,
    #[note(
        "certain types, like `{$ty}`, must be cast before passing them to a variadic function to match the implicit cast that a C compiler would perform as part of C's numeric promotion rules"
    )]
    pub(crate) teach: bool,
}

#[derive(Diagnostic)]
#[diag("can't pass a function item to a variadic function", code = E0617)]
#[help(
    "a function item is zero-sized and needs to be cast into a function pointer to be used in FFI"
)]
#[note(
    "for more information on function items, visit https://doc.rust-lang.org/reference/types/function-item.html"
)]
pub(crate) struct PassFnItemToVariadicFunction {
    #[primary_span]
    pub span: Span,
    #[suggestion(
        "use a function pointer instead",
        code = " as {replace}",
        applicability = "machine-applicable",
        style = "verbose"
    )]
    pub sugg_span: Span,
    pub replace: String,
}

#[derive(Subdiagnostic)]
#[suggestion(
    "replace the comma with a semicolon to create {$descr}",
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
#[diag("trait item `{$item}` from `{$subtrait}` shadows identically named item from supertrait")]
pub(crate) struct SupertraitItemShadowing {
    pub item: Symbol,
    pub subtrait: Symbol,
    #[subdiagnostic]
    pub shadower: SupertraitItemShadower,
    #[subdiagnostic]
    pub shadowee: SupertraitItemShadowee,
}

#[derive(Subdiagnostic)]
#[note("item from `{$subtrait}` shadows a supertrait item")]
pub(crate) struct SupertraitItemShadower {
    pub subtrait: Symbol,
    #[primary_span]
    pub span: Span,
}

#[derive(Subdiagnostic)]
pub(crate) enum SupertraitItemShadowee {
    #[note("item from `{$supertrait}` is shadowed by a subtrait item")]
    Labeled {
        #[primary_span]
        span: Span,
        supertrait: Symbol,
    },
    #[note("items from several supertraits are shadowed: {$traits}")]
    Several {
        #[primary_span]
        spans: MultiSpan,
        traits: DiagSymbolList,
    },
}

#[derive(Diagnostic)]
#[diag("type `{$ty}` cannot be used with this register class in stable")]
pub(crate) struct RegisterTypeUnstable<'a> {
    #[primary_span]
    pub span: Span,
    pub ty: Ty<'a>,
}

#[derive(Diagnostic)]
#[diag("the `naked_asm!` macro can only be used in functions marked with `#[unsafe(naked)]`")]
pub(crate) struct NakedAsmOutsideNakedFn {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("patterns not allowed in naked function parameters")]
pub(crate) struct NoPatterns {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("referencing function parameters is not allowed in naked functions")]
#[help("follow the calling convention in asm block to use parameters")]
pub(crate) struct ParamsNotAllowed {
    #[primary_span]
    pub span: Span,
}

pub(crate) struct NakedFunctionsAsmBlock {
    pub span: Span,
    pub multiple_asms: Vec<Span>,
    pub non_asms: Vec<Span>,
}

impl<G: EmissionGuarantee> Diagnostic<'_, G> for NakedFunctionsAsmBlock {
    #[track_caller]
    fn into_diag(self, dcx: DiagCtxtHandle<'_>, level: Level) -> Diag<'_, G> {
        let mut diag = Diag::new(
            dcx,
            level,
            msg!("naked functions must contain a single `naked_asm!` invocation"),
        );
        diag.span(self.span);
        diag.code(E0787);
        for span in self.multiple_asms.iter() {
            diag.span_label(
                *span,
                msg!("multiple `naked_asm!` invocations are not allowed in naked functions"),
            );
        }
        for span in self.non_asms.iter() {
            diag.span_label(*span, msg!("not allowed in naked functions"));
        }
        diag
    }
}

pub(crate) fn maybe_emit_plus_equals_diagnostic<'a>(
    fnctxt: &FnCtxt<'a, '_>,
    assign_op: Spanned<AssignOpKind>,
    lhs_expr: &hir::Expr<'_>,
) -> Result<(), Diag<'a>> {
    if assign_op.node == hir::AssignOpKind::AddAssign
        && let hir::ExprKind::Binary(bin_op, left, right) = &lhs_expr.kind
        && bin_op.node == hir::BinOpKind::And
        && crate::op::contains_let_in_chain(left)
        && let hir::ExprKind::Path(hir::QPath::Resolved(_, path)) = &right.kind
        && matches!(path.res, hir::def::Res::Local(_))
    {
        let mut err = fnctxt.dcx().struct_span_err(
            assign_op.span,
            "binary assignment operation `+=` cannot be used in a let chain",
        );

        err.span_label(
            lhs_expr.span,
            "you are add-assigning the right-hand side expression to the result of this let-chain",
        );

        err.span_label(assign_op.span, "cannot use `+=` in a let chain");

        err.span_suggestion(
            assign_op.span,
            "you might have meant to compare with `==` instead of assigning with `+=`",
            "==",
            Applicability::MaybeIncorrect,
        );

        return Err(err);
    }
    Ok(())
}

#[derive(Diagnostic)]
#[diag("the `asm!` macro is not allowed in naked functions", code = E0787)]
pub(crate) struct NakedFunctionsMustNakedAsm {
    #[primary_span]
    #[label("consider using the `naked_asm!` macro instead")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("functions with the {$abi} ABI cannot be called")]
pub(crate) struct AbiCannotBeCalled {
    #[primary_span]
    #[note("an `extern {$abi}` function can only be called using inline assembly")]
    pub span: Span,
    pub abi: ExternAbi,
}

#[derive(Diagnostic)]
#[diag("functions with the \"gpu-kernel\" ABI cannot be called")]
pub(crate) struct GpuKernelAbiCannotBeCalled {
    #[primary_span]
    #[note("an `extern \"gpu-kernel\"` function must be launched on the GPU by the runtime")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("`#[const_continue]` must break to a labeled block that participates in a `#[loop_match]`")]
pub(crate) struct ConstContinueBadLabel {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("cannot project on type that is not `#[pin_v2]`")]
pub(crate) struct ProjectOnNonPinProjectType {
    #[primary_span]
    pub span: Span,
    #[note("type defined here")]
    pub def_span: Option<Span>,
    #[suggestion(
        "add `#[pin_v2]` here",
        code = "#[pin_v2]\n",
        applicability = "machine-applicable"
    )]
    pub sugg_span: Option<Span>,
}
