use rustc_errors::{fluent, AddSubdiagnostic, Applicability, DecorateLint, EmissionGuarantee};
use rustc_hir::def_id::DefId;
use rustc_macros::{LintDiagnostic, SessionSubdiagnostic};
use rustc_middle::ty::{Predicate, Ty, TyCtxt};
use rustc_span::{symbol::Ident, Span, Symbol};

use crate::LateContext;

pub struct NonFmtPanicUnused {
    pub count: usize,
    pub suggestion: Option<Span>,
}

impl<G: EmissionGuarantee> DecorateLint<'_, G> for NonFmtPanicUnused {
    fn decorate_lint(self, diag: rustc_errors::LintDiagnosticBuilder<'_, G>) {
        let mut diag = diag.build(fluent::lint_non_fmt_panic_unused);
        diag.set_arg("count", self.count);
        diag.note(fluent::note);
        if let Some(span) = self.suggestion {
            diag.span_suggestion(
                span.shrink_to_hi(),
                fluent::add_args_suggestion,
                ", ...",
                Applicability::HasPlaceholders,
            );
            diag.span_suggestion(
                span.shrink_to_lo(),
                fluent::add_fmt_suggestion,
                "\"{}\", ",
                Applicability::MachineApplicable,
            );
        }
        diag.emit();
    }
}

#[derive(LintDiagnostic)]
#[diag(lint_non_fmt_panic_braces)]
#[note]
pub struct NonFmtPanicBraces {
    pub count: usize,
    #[suggestion(code = "\"{{}}\", ", applicability = "machine-applicable")]
    pub suggestion: Option<Span>,
}

#[derive(LintDiagnostic)]
#[diag(lint_non_camel_case_type)]
pub struct NonCamelCaseType<'a> {
    pub sort: &'a str,
    pub name: &'a str,
    #[subdiagnostic]
    pub sub: NonCamelCaseTypeSub,
}

#[derive(SessionSubdiagnostic)]
pub enum NonCamelCaseTypeSub {
    #[label(label)]
    Label {
        #[primary_span]
        span: Span,
    },
    #[suggestion(suggestion, code = "{replace}", applicability = "maybe-incorrect")]
    Suggestion {
        #[primary_span]
        span: Span,
        replace: String,
    },
}

#[derive(LintDiagnostic)]
#[diag(lint_non_snake_case)]
pub struct NonSnakeCaseDiag<'a> {
    pub sort: &'a str,
    pub name: &'a str,
    pub sc: String,
    #[subdiagnostic]
    pub sub: NonSnakeCaseDiagSub,
}

pub enum NonSnakeCaseDiagSub {
    Label { span: Span },
    Help,
    RenameOrConvertSuggestion { span: Span, suggestion: Ident },
    ConvertSuggestion { span: Span, suggestion: String },
    SuggestionAndNote { span: Span },
}

impl AddSubdiagnostic for NonSnakeCaseDiagSub {
    fn add_to_diagnostic(self, diag: &mut rustc_errors::Diagnostic) {
        match self {
            NonSnakeCaseDiagSub::Label { span } => {
                diag.span_label(span, fluent::label);
            }
            NonSnakeCaseDiagSub::Help => {
                diag.help(fluent::help);
            }
            NonSnakeCaseDiagSub::ConvertSuggestion { span, suggestion } => {
                diag.span_suggestion(
                    span,
                    fluent::convert_suggestion,
                    suggestion,
                    Applicability::MaybeIncorrect,
                );
            }
            NonSnakeCaseDiagSub::RenameOrConvertSuggestion { span, suggestion } => {
                diag.span_suggestion(
                    span,
                    fluent::rename_or_convert_suggestion,
                    suggestion,
                    Applicability::MaybeIncorrect,
                );
            }
            NonSnakeCaseDiagSub::SuggestionAndNote { span } => {
                diag.note(fluent::cannot_convert_note);
                diag.span_suggestion(
                    span,
                    fluent::rename_suggestion,
                    "",
                    Applicability::MaybeIncorrect,
                );
            }
        };
    }
}

#[derive(LintDiagnostic)]
#[diag(lint_non_upper_case_global)]
pub struct NonUpperCaseGlobal<'a> {
    pub sort: &'a str,
    pub name: &'a str,
    #[subdiagnostic]
    pub sub: NonUpperCaseGlobalSub,
}

#[derive(SessionSubdiagnostic)]
pub enum NonUpperCaseGlobalSub {
    #[label(label)]
    Label {
        #[primary_span]
        span: Span,
    },
    #[suggestion(suggestion, code = "{replace}", applicability = "maybe-incorrect")]
    Suggestion {
        #[primary_span]
        span: Span,
        replace: String,
    },
}

#[derive(LintDiagnostic)]
#[diag(lint_noop_method_call)]
#[note]
pub struct NoopMethodCallDiag<'a> {
    pub method: Symbol,
    pub receiver_ty: Ty<'a>,
    #[label]
    pub label: Span,
}

#[derive(LintDiagnostic)]
#[diag(lint_pass_by_value)]
pub struct PassByValueDiag {
    pub ty: String,
    #[suggestion(code = "{ty}", applicability = "maybe-incorrect")]
    pub suggestion: Span,
}

#[derive(LintDiagnostic)]
#[diag(lint_redundant_semicolons)]
pub struct RedundantSemicolonsDiag {
    pub multiple: bool,
    #[suggestion(code = "", applicability = "maybe-incorrect")]
    pub suggestion: Span,
}

pub struct DropTraitConstraintsDiag<'a> {
    pub predicate: Predicate<'a>,
    pub tcx: TyCtxt<'a>,
    pub def_id: DefId,
}

impl<'a, G: EmissionGuarantee> DecorateLint<'_, G> for DropTraitConstraintsDiag<'a> {
    fn decorate_lint(self, diag: rustc_errors::LintDiagnosticBuilder<'_, G>) {
        let mut diag = diag.build(fluent::lint_drop_trait_constraints);
        diag.set_arg("predicate", self.predicate);
        diag.set_arg("needs_drop", self.tcx.def_path_str(self.def_id));
        diag.emit();
    }
}

pub struct DropGlue<'a> {
    pub tcx: TyCtxt<'a>,
    pub def_id: DefId,
}

impl<'a, G: EmissionGuarantee> DecorateLint<'_, G> for DropGlue<'a> {
    fn decorate_lint(self, diag: rustc_errors::LintDiagnosticBuilder<'_, G>) {
        let mut diag = diag.build(fluent::lint_drop_glue);
        diag.set_arg("needs_drop", self.tcx.def_path_str(self.def_id));
        diag.emit();
    }
}

#[derive(LintDiagnostic)]
#[diag(lint_range_endpoint_out_of_range)]
pub struct RangeEndpointOutOfRange<'a> {
    pub ty: &'a str,
    #[suggestion(code = "{start}..={literal}{suffix}", applicability = "machine-applicable")]
    pub suggestion: Span,
    pub start: String,
    pub literal: u128,
    pub suffix: &'a str,
}

#[derive(LintDiagnostic)]
#[diag(lint_overflowing_bin_hex)]
pub struct OverflowingBinHex<'a> {
    pub ty: &'a str,
    pub lit: String,
    pub dec: u128,
    pub actually: String,
    #[subdiagnostic]
    pub sign: OverflowingBinHexSign,
    #[subdiagnostic]
    pub sub: Option<OverflowingBinHexSub<'a>>,
}

pub enum OverflowingBinHexSign {
    Positive,
    Negative,
}

impl AddSubdiagnostic for OverflowingBinHexSign {
    fn add_to_diagnostic(self, diag: &mut rustc_errors::Diagnostic) {
        match self {
            OverflowingBinHexSign::Positive => {
                diag.note(fluent::positive_note);
            }
            OverflowingBinHexSign::Negative => {
                diag.note(fluent::negative_note);
                diag.note(fluent::negative_becomes_note);
            }
        }
    }
}

#[derive(SessionSubdiagnostic)]
pub enum OverflowingBinHexSub<'a> {
    #[suggestion(
        suggestion,
        code = "{sans_suffix}{suggestion_ty}",
        applicability = "machine-applicable"
    )]
    Suggestion {
        #[primary_span]
        span: Span,
        suggestion_ty: &'a str,
        sans_suffix: &'a str,
    },
    #[help(help)]
    Help { suggestion_ty: &'a str },
}

pub struct OverflowingInt<'a> {
    pub ty: &'a str,
    pub lit: String,
    pub min: i128,
    pub max: u128,
    pub suggestion_ty: Option<&'a str>,
}

// FIXME: refactor with `Option<&'a str>` in macro
impl<'a, G: EmissionGuarantee> DecorateLint<'_, G> for OverflowingInt<'a> {
    fn decorate_lint(self, diag: rustc_errors::LintDiagnosticBuilder<'_, G>) {
        let mut diag = diag.build(fluent::lint_overflowing_int);
        diag.set_arg("ty", self.ty);
        diag.set_arg("lit", self.lit);
        diag.set_arg("min", self.min);
        diag.set_arg("max", self.max);
        diag.note(fluent::note);
        if let Some(suggestion_ty) = self.suggestion_ty {
            diag.set_arg("suggestion_ty", suggestion_ty);
            diag.help(fluent::help);
        }
        diag.emit();
    }
}

#[derive(LintDiagnostic)]
#[diag(lint_only_cast_u8_to_char)]
pub struct OnlyCastu8ToChar {
    #[suggestion(code = "'\\u{{{literal:X}}}'", applicability = "machine-applicable")]
    pub span: Span,
    pub literal: u128,
}

#[derive(LintDiagnostic)]
#[diag(lint_overflowing_uint)]
#[note]
pub struct OverflowingUInt<'a> {
    pub ty: &'a str,
    pub lit: String,
    pub min: u128,
    pub max: u128,
}

#[derive(LintDiagnostic)]
#[diag(lint_overflowing_literal)]
#[note]
pub struct OverflowingLiteral<'a> {
    pub ty: &'a str,
    pub lit: String,
}

#[derive(LintDiagnostic)]
#[diag(lint_unused_comparisons)]
pub struct UnusedComparisons;

#[derive(LintDiagnostic)]
#[diag(lint_variant_size_differences)]
pub struct VariantSizeDifferencesDiag {
    pub largest: u64,
}

#[derive(LintDiagnostic)]
#[diag(lint_atomic_ordering_load)]
#[help]
pub struct AtomicOrderingLoad;

#[derive(LintDiagnostic)]
#[diag(lint_atomic_ordering_store)]
#[help]
pub struct AtomicOrderingStore;

#[derive(LintDiagnostic)]
#[diag(lint_atomic_ordering_fence)]
#[help]
pub struct AtomicOrderingFence;

#[derive(LintDiagnostic)]
#[diag(lint_atomic_ordering_invalid)]
#[help]
pub struct InvalidAtomicOrderingDiag {
    pub method: Symbol,
    #[label]
    pub fail_order_arg_span: Span,
}

#[derive(LintDiagnostic)]
#[diag(lint_unused_op)]
pub struct UnusedOp<'a> {
    pub op: &'a str,
    #[label]
    pub label: Span,
    #[suggestion(style = "verbose", code = "let _ = ", applicability = "machine-applicable")]
    pub suggestion: Span,
}

#[derive(LintDiagnostic)]
#[diag(lint_unused_result)]
pub struct UnusedResult<'a> {
    pub ty: Ty<'a>,
}

// FIXME(davidtwco): this isn't properly translatable becauses of the
// pre/post strings
#[derive(LintDiagnostic)]
#[diag(lint_unused_closure)]
#[note]
pub struct UnusedClosure<'a> {
    pub count: usize,
    pub pre: &'a str,
    pub post: &'a str,
}

// FIXME(davidtwco): this isn't properly translatable becauses of the
// pre/post strings
#[derive(LintDiagnostic)]
#[diag(lint_unused_generator)]
#[note]
pub struct UnusedGenerator<'a> {
    pub count: usize,
    pub pre: &'a str,
    pub post: &'a str,
}

// FIXME(davidtwco): this isn't properly translatable becauses of the pre/post
// strings
pub struct UnusedDef<'a, 'b> {
    pub pre: &'a str,
    pub post: &'a str,
    pub cx: &'a LateContext<'b>,
    pub def_id: DefId,
    pub note: Option<Symbol>,
}

// FIXME: refactor with `Option<String>` in macro
impl<'a, 'b, G: EmissionGuarantee> DecorateLint<'_, G> for UnusedDef<'a, 'b> {
    fn decorate_lint(self, diag: rustc_errors::LintDiagnosticBuilder<'_, G>) {
        let mut diag = diag.build(fluent::lint_unused_def);
        diag.set_arg("pre", self.pre);
        diag.set_arg("post", self.post);
        diag.set_arg("def", self.cx.tcx.def_path_str(self.def_id));
        // check for #[must_use = "..."]
        if let Some(note) = self.note {
            diag.note(note.as_str());
        }
        diag.emit();
    }
}

#[derive(LintDiagnostic)]
#[diag(lint_path_statement_drop)]
pub struct PathStatementDrop {
    #[subdiagnostic]
    pub sub: PathStatementDropSub,
}

#[derive(SessionSubdiagnostic)]
pub enum PathStatementDropSub {
    #[suggestion(
        suggestion,
        code = "drop({snippet});",
        applicability = "machine-applicable"
    )]
    Suggestion {
        #[primary_span]
        span: Span,
        snippet: String,
    },
    #[help(help)]
    Help {
        #[primary_span]
        span: Span,
    },
}

#[derive(LintDiagnostic)]
#[diag(lint_path_statement_no_effect)]
pub struct PathStatementNoEffect;

#[derive(LintDiagnostic)]
#[diag(lint_unused_delim)]
pub struct UnusedDelim<'a> {
    pub delim: &'static str,
    pub item: &'a str,
    #[subdiagnostic]
    pub suggestion: Option<UnusedDelimSuggestion>,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(suggestion, applicability = "machine-applicable")]
pub struct UnusedDelimSuggestion {
    #[suggestion_part(code = "{start_replace}")]
    pub start_span: Span,
    pub start_replace: &'static str,
    #[suggestion_part(code = "{end_replace}")]
    pub end_span: Span,
    pub end_replace: &'static str,
}

#[derive(LintDiagnostic)]
#[diag(lint_unused_import_braces)]
pub struct UnusedImportBracesDiag {
    pub node: Symbol,
}

#[derive(LintDiagnostic)]
#[diag(lint_unused_allocation)]
pub struct UnusedAllocationDiag;

#[derive(LintDiagnostic)]
#[diag(lint_unused_allocation_mut)]
pub struct UnusedAllocationMutDiag;
