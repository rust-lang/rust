use std::num::NonZeroU32;

use rustc_errors::{fluent, AddToDiagnostic, Applicability, DecorateLint, DiagnosticMessage};
use rustc_hir::def_id::DefId;
use rustc_macros::{LintDiagnostic, Subdiagnostic};
use rustc_middle::{
    lint::LintExpectation,
    ty::{Predicate, Ty, TyCtxt},
};
use rustc_span::{edition::Edition, symbol::Ident, Span, Symbol};

use crate::{errors::OverruledAttributeSub, LateContext};

// array_into_iter.rs
#[derive(LintDiagnostic)]
#[diag(lint_array_into_iter)]
pub struct ArrayIntoIterDiag<'a> {
    pub target: &'a str,
    #[suggestion(use_iter_suggestion, code = "iter", applicability = "machine-applicable")]
    pub suggestion: Span,
    #[subdiagnostic]
    pub sub: Option<ArrayIntoIterDiagSub>,
}

#[derive(Subdiagnostic)]
pub enum ArrayIntoIterDiagSub {
    #[suggestion(remove_into_iter_suggestion, code = "", applicability = "maybe-incorrect")]
    RemoveIntoIter {
        #[primary_span]
        span: Span,
    },
    #[multipart_suggestion(use_explicit_into_iter_suggestion, applicability = "maybe-incorrect")]
    UseExplicitIntoIter {
        #[suggestion_part(code = "IntoIterator::into_iter(")]
        start_span: Span,
        #[suggestion_part(code = ")")]
        end_span: Span,
    },
}

// builtin.rs
#[derive(LintDiagnostic)]
#[diag(lint_builtin_while_true)]
pub struct BuiltinWhileTrue {
    #[suggestion(style = "short", code = "{replace}", applicability = "machine-applicable")]
    pub suggestion: Span,
    pub replace: String,
}

#[derive(LintDiagnostic)]
#[diag(lint_builtin_box_pointers)]
pub struct BuiltinBoxPointers<'a> {
    pub ty: Ty<'a>,
}

#[derive(LintDiagnostic)]
#[diag(lint_builtin_non_shorthand_field_patterns)]
pub struct BuiltinNonShorthandFieldPatterns {
    pub ident: Ident,
    #[suggestion(code = "{prefix}{ident}", applicability = "machine-applicable")]
    pub suggestion: Span,
    pub prefix: &'static str,
}

// FIXME: add lint::unsafe_code

#[derive(LintDiagnostic)]
#[diag(lint_builtin_missing_doc)]
pub struct BuiltinMissingDoc<'a> {
    pub article: &'a str,
    pub desc: &'a str,
}

#[derive(LintDiagnostic)]
#[diag(lint_builtin_missing_copy_impl)]
pub struct BuiltinMissingCopyImpl;

pub struct BuiltinMissingDebugImpl<'a> {
    pub tcx: TyCtxt<'a>,
    pub def_id: DefId,
}

impl<'a> DecorateLint<'a, ()> for BuiltinMissingDebugImpl<'_> {
    fn decorate_lint<'b>(
        self,
        diag: &'b mut rustc_errors::DiagnosticBuilder<'a, ()>,
    ) -> &'b mut rustc_errors::DiagnosticBuilder<'a, ()> {
        diag.set_arg("debug", self.tcx.def_path_str(self.def_id));
        diag
    }

    fn msg(&self) -> DiagnosticMessage {
        fluent::lint_builtin_missing_debug_impl
    }
}

#[derive(LintDiagnostic)]
#[diag(lint_builtin_anonymous_params)]
pub struct BuiltinAnonymousParams<'a> {
    #[suggestion(code = "_: {ty_snip}")]
    pub suggestion: (Span, Applicability),
    pub ty_snip: &'a str,
}

// FIXME: add lint::builtin_deprecated_attr_link

#[derive(LintDiagnostic)]
#[diag(lint_builtin_deprecated_attr_used)]
pub struct BuiltinDeprecatedAttrUsed {
    pub name: String,
    #[suggestion(
        lint_builtin_deprecated_attr_default_suggestion,
        style = "short",
        code = "",
        applicability = "machine-applicable"
    )]
    pub suggestion: Span,
}

#[derive(LintDiagnostic)]
#[diag(lint_builtin_unused_doc_comment)]
pub struct BuiltinUnusedDocComment<'a> {
    pub kind: &'a str,
    #[label]
    pub label: Span,
    #[subdiagnostic]
    pub sub: BuiltinUnusedDocCommentSub,
}

#[derive(Subdiagnostic)]
pub enum BuiltinUnusedDocCommentSub {
    #[help(plain_help)]
    PlainHelp,
    #[help(block_help)]
    BlockHelp,
}

#[derive(LintDiagnostic)]
#[diag(lint_builtin_no_mangle_generic)]
pub struct BuiltinNoMangleGeneric {
    // Use of `#[no_mangle]` suggests FFI intent; correct
    // fix may be to monomorphize source by hand
    #[suggestion(style = "short", code = "", applicability = "maybe-incorrect")]
    pub suggestion: Span,
}

#[derive(LintDiagnostic)]
#[diag(lint_builtin_const_no_mangle)]
pub struct BuiltinConstNoMangle {
    #[suggestion(code = "pub static", applicability = "machine-applicable")]
    pub suggestion: Span,
}

#[derive(LintDiagnostic)]
#[diag(lint_builtin_mutable_transmutes)]
pub struct BuiltinMutablesTransmutes;

#[derive(LintDiagnostic)]
#[diag(lint_builtin_unstable_features)]
pub struct BuiltinUnstableFeatures;

#[derive(LintDiagnostic)]
#[diag(lint_builtin_unreachable_pub)]
pub struct BuiltinUnreachablePub<'a> {
    pub what: &'a str,
    #[suggestion(code = "pub(crate)")]
    pub suggestion: (Span, Applicability),
    #[help]
    pub help: Option<()>,
}

// FIXME: migrate builtin_type_alias_where_clause

// FIXME: migrate builtin_type_alias_generic_bounds

#[derive(LintDiagnostic)]
#[diag(lint_builtin_trivial_bounds)]
pub struct BuiltinTrivialBounds<'a> {
    pub predicate_kind_name: &'a str,
    pub predicate: Predicate<'a>,
}

#[derive(LintDiagnostic)]
pub enum BuiltinEllipsisInclusiveRangePatternsLint {
    #[diag(lint_builtin_ellipsis_inclusive_range_patterns)]
    Parenthesise {
        #[suggestion(code = "{replace}", applicability = "machine-applicable")]
        suggestion: Span,
        replace: String,
    },
    #[diag(lint_builtin_ellipsis_inclusive_range_patterns)]
    NonParenthesise {
        #[suggestion(style = "short", code = "..=", applicability = "machine-applicable")]
        suggestion: Span,
    },
}

#[derive(LintDiagnostic)]
#[diag(lint_builtin_unnameable_test_items)]
pub struct BuiltinUnnameableTestItems;

#[derive(LintDiagnostic)]
#[diag(lint_builtin_keyword_idents)]
pub struct BuiltinKeywordIdents {
    pub kw: Ident,
    pub next: Edition,
    #[suggestion(code = "r#{kw}", applicability = "machine-applicable")]
    pub suggestion: Span,
}

#[derive(LintDiagnostic)]
#[diag(lint_builtin_explicit_outlives)]
pub struct BuiltinExplicitOutlives {
    pub count: usize,
    #[subdiagnostic]
    pub suggestion: BuiltinExplicitOutlivesSuggestion,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(suggestion)]
pub struct BuiltinExplicitOutlivesSuggestion {
    #[suggestion_part(code = "")]
    pub spans: Vec<Span>,
    #[applicability]
    pub applicability: Applicability,
}

pub struct BuiltinIncompleteFeatures {
    pub name: Symbol,
    pub note: Option<NonZeroU32>,
    pub help: Option<()>,
}

impl<'a> DecorateLint<'a, ()> for BuiltinIncompleteFeatures {
    fn decorate_lint<'b>(
        self,
        diag: &'b mut rustc_errors::DiagnosticBuilder<'a, ()>,
    ) -> &'b mut rustc_errors::DiagnosticBuilder<'a, ()> {
        diag.set_arg("name", self.name);
        if let Some(n) = self.note {
            diag.set_arg("n", n);
            diag.note(fluent::note);
        }
        if let Some(_) = self.help {
            diag.help(fluent::help);
        }
        diag
    }

    fn msg(&self) -> DiagnosticMessage {
        fluent::lint_builtin_incomplete_features
    }
}

// FIXME: migrate "the type `{}` does not permit {}"

// FIXME: fluent::lint::builtin_clashing_extern_{same,diff}_name

#[derive(LintDiagnostic)]
#[diag(lint_builtin_deref_nullptr)]
pub struct BuiltinDerefNullptr {
    #[label]
    pub label: Span,
}

// FIXME: migrate fluent::lint::builtin_asm_labels

#[derive(LintDiagnostic)]
pub enum BuiltinSpecialModuleNameUsed {
    #[diag(lint_builtin_special_module_name_used_lib)]
    #[note]
    #[help]
    Lib,
    #[diag(lint_builtin_special_module_name_used_main)]
    #[note]
    Main,
}

#[derive(LintDiagnostic)]
#[diag(lint_builtin_unexpected_cli_config_name)]
#[help]
pub struct BuiltinUnexpectedCliConfigName {
    pub name: Symbol,
}

#[derive(LintDiagnostic)]
#[diag(lint_builtin_unexpected_cli_config_value)]
#[help]
pub struct BuiltinUnexpectedCliConfigValue {
    pub name: Symbol,
    pub value: Symbol,
}

// enum_intrinsics_non_enums.rs
#[derive(LintDiagnostic)]
#[diag(lint_enum_intrinsics_mem_discriminant)]
pub struct EnumIntrinsicsMemDiscriminate<'a> {
    pub ty_param: Ty<'a>,
    #[note]
    pub note: Span,
}

#[derive(LintDiagnostic)]
#[diag(lint_enum_intrinsics_mem_variant)]
#[note]
pub struct EnumIntrinsicsMemVariant<'a> {
    pub ty_param: Ty<'a>,
}

// expect.rs
pub struct Expectation<'a> {
    pub expectation: &'a LintExpectation,
}

impl<'a> DecorateLint<'a, ()> for Expectation<'_> {
    fn decorate_lint<'b>(
        self,
        diag: &'b mut rustc_errors::DiagnosticBuilder<'a, ()>,
    ) -> &'b mut rustc_errors::DiagnosticBuilder<'a, ()> {
        if let Some(rationale) = self.expectation.reason {
            diag.note(rationale.as_str());
        }

        if self.expectation.is_unfulfilled_lint_expectations {
            diag.note(fluent::note);
        }

        diag
    }

    fn msg(&self) -> DiagnosticMessage {
        fluent::lint_expectation
    }
}

// for_loops_over_fallibles.rs
#[derive(LintDiagnostic)]
#[diag(lint_for_loops_over_fallibles)]
pub struct ForLoopsOverFalliblesDiag<'a> {
    pub article: &'static str,
    pub ty: &'static str,
    #[subdiagnostic]
    pub sub: ForLoopsOverFalliblesLoopSub<'a>,
    #[subdiagnostic]
    pub question_mark: Option<ForLoopsOverFalliblesQuestionMark>,
    #[subdiagnostic]
    pub suggestion: ForLoopsOverFalliblesSuggestion<'a>,
}

#[derive(Subdiagnostic)]
pub enum ForLoopsOverFalliblesLoopSub<'a> {
    #[suggestion(remove_next, code = ".by_ref()", applicability = "maybe-incorrect")]
    RemoveNext {
        #[primary_span]
        suggestion: Span,
        recv_snip: String,
    },
    #[multipart_suggestion(use_while_let, applicability = "maybe-incorrect")]
    UseWhileLet {
        #[suggestion_part(code = "while let {var}(")]
        start_span: Span,
        #[suggestion_part(code = ") = ")]
        end_span: Span,
        var: &'a str,
    },
}

#[derive(Subdiagnostic)]
#[suggestion(use_question_mark, code = "?", applicability = "maybe-incorrect")]
pub struct ForLoopsOverFalliblesQuestionMark {
    #[primary_span]
    pub suggestion: Span,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(suggestion, applicability = "maybe-incorrect")]
pub struct ForLoopsOverFalliblesSuggestion<'a> {
    pub var: &'a str,
    #[suggestion_part(code = "if let {var}(")]
    pub start_span: Span,
    #[suggestion_part(code = ") = ")]
    pub end_span: Span,
}

// internal.rs
#[derive(LintDiagnostic)]
#[diag(lint_default_hash_types)]
#[note]
pub struct DefaultHashTypesDiag<'a> {
    pub preferred: &'a str,
    pub used: Symbol,
}

#[derive(LintDiagnostic)]
#[diag(lint_query_instability)]
#[note]
pub struct QueryInstability {
    pub query: Symbol,
}

#[derive(LintDiagnostic)]
#[diag(lint_tykind_kind)]
pub struct TykindKind {
    #[suggestion(code = "ty", applicability = "maybe-incorrect")]
    pub suggestion: Span,
}

#[derive(LintDiagnostic)]
#[diag(lint_tykind)]
#[help]
pub struct TykindDiag;

#[derive(LintDiagnostic)]
#[diag(lint_ty_qualified)]
pub struct TyQualified {
    pub ty: String,
    #[suggestion(code = "{ty}", applicability = "maybe-incorrect")]
    pub suggestion: Span,
}

#[derive(LintDiagnostic)]
#[diag(lint_lintpass_by_hand)]
#[help]
pub struct LintPassByHand;

#[derive(LintDiagnostic)]
#[diag(lint_non_existant_doc_keyword)]
#[help]
pub struct NonExistantDocKeyword {
    pub keyword: Symbol,
}

#[derive(LintDiagnostic)]
#[diag(lint_diag_out_of_impl)]
pub struct DiagOutOfImpl;

#[derive(LintDiagnostic)]
#[diag(lint_untranslatable_diag)]
pub struct UntranslatableDiag;

#[derive(LintDiagnostic)]
#[diag(lint_bad_opt_access)]
pub struct BadOptAccessDiag<'a> {
    pub msg: &'a str,
}

// let_underscore.rs
#[derive(LintDiagnostic)]
pub enum NonBindingLet {
    #[diag(lint_non_binding_let_on_sync_lock)]
    SyncLock {
        #[subdiagnostic]
        sub: NonBindingLetSub,
    },
    #[diag(lint_non_binding_let_on_drop_type)]
    DropType {
        #[subdiagnostic]
        sub: NonBindingLetSub,
    },
}

pub struct NonBindingLetSub {
    pub suggestion: Span,
    pub multi_suggestion_start: Span,
    pub multi_suggestion_end: Span,
}

impl AddToDiagnostic for NonBindingLetSub {
    fn add_to_diagnostic_with<F>(self, diag: &mut rustc_errors::Diagnostic, _: F)
    where
        F: Fn(
            &mut rustc_errors::Diagnostic,
            rustc_errors::SubdiagnosticMessage,
        ) -> rustc_errors::SubdiagnosticMessage,
    {
        diag.span_suggestion_verbose(
            self.suggestion,
            fluent::lint_non_binding_let_suggestion,
            "_unused",
            Applicability::MachineApplicable,
        );
        diag.multipart_suggestion(
            fluent::lint_non_binding_let_multi_suggestion,
            vec![
                (self.multi_suggestion_start, "drop(".to_string()),
                (self.multi_suggestion_end, ")".to_string()),
            ],
            Applicability::MachineApplicable,
        );
    }
}

// levels.rs
#[derive(LintDiagnostic)]
#[diag(lint_overruled_attribute)]
pub struct OverruledAtributeLint<'a> {
    #[label]
    pub overruled: Span,
    pub lint_level: &'a str,
    pub lint_source: Symbol,
    #[subdiagnostic]
    pub sub: OverruledAttributeSub,
}

#[derive(LintDiagnostic)]
#[diag(lint_deprecated_lint_name)]
pub struct DeprecatedLintName<'a> {
    pub name: String,
    #[suggestion(code = "{replace}", applicability = "machine-applicable")]
    pub suggestion: Span,
    pub replace: &'a str,
}

pub struct RenamedOrRemovedLint<'a> {
    pub msg: &'a str,
    pub suggestion: Span,
    pub renamed: &'a Option<String>,
}

impl<'a> DecorateLint<'a, ()> for RenamedOrRemovedLint<'_> {
    fn decorate_lint<'b>(
        self,
        diag: &'b mut rustc_errors::DiagnosticBuilder<'a, ()>,
    ) -> &'b mut rustc_errors::DiagnosticBuilder<'a, ()> {
        if let Some(new_name) = self.renamed {
            diag.span_suggestion(
                self.suggestion,
                fluent::lint_renamed_or_removed_lint_suggestion,
                new_name,
                Applicability::MachineApplicable,
            );
        };
        diag
    }

    fn msg(&self) -> rustc_errors::DiagnosticMessage {
        rustc_errors::DiagnosticMessage::Str(self.msg.to_string())
    }
}

pub struct UnknownLint<'a> {
    pub name: String,
    pub suggestion: Span,
    pub replace: &'a Option<Symbol>,
}

impl<'a> DecorateLint<'a, ()> for UnknownLint<'_> {
    fn decorate_lint<'b>(
        self,
        diag: &'b mut rustc_errors::DiagnosticBuilder<'a, ()>,
    ) -> &'b mut rustc_errors::DiagnosticBuilder<'a, ()> {
        diag.set_arg("name", self.name);
        if let Some(replace) = self.replace {
            diag.span_suggestion(
                self.suggestion,
                fluent::suggestion,
                replace,
                Applicability::MaybeIncorrect,
            );
        };
        diag
    }

    fn msg(&self) -> rustc_errors::DiagnosticMessage {
        fluent::lint_unknown_lint
    }
}

#[derive(LintDiagnostic)]
#[diag(lint_ignored_unless_crate_specified)]
pub struct IgnoredUnlessCrateSpecified<'a> {
    pub level: &'a str,
    pub name: Symbol,
}

// methods.rs
#[derive(LintDiagnostic)]
#[diag(lint_cstring_ptr)]
#[note]
#[help]
pub struct CStringPtr {
    #[label(as_ptr_label)]
    pub as_ptr: Span,
    #[label(unwrap_label)]
    pub unwrap: Span,
}

// non_ascii_idents.rs
#[derive(LintDiagnostic)]
#[diag(lint_identifier_non_ascii_char)]
pub struct IdentifierNonAsciiChar;

#[derive(LintDiagnostic)]
#[diag(lint_identifier_uncommon_codepoints)]
pub struct IdentifierUncommonCodepoints;

#[derive(LintDiagnostic)]
#[diag(lint_confusable_identifier_pair)]
pub struct ConfusableIdentifierPair {
    pub existing_sym: Symbol,
    pub sym: Symbol,
    #[label]
    pub label: Span,
}

#[derive(LintDiagnostic)]
#[diag(lint_mixed_script_confusables)]
#[note(includes_note)]
#[note]
pub struct MixedScriptConfusables {
    pub set: String,
    pub includes: String,
}

// non_fmt_panic.rs
pub struct NonFmtPanicUnused {
    pub count: usize,
    pub suggestion: Option<Span>,
}

impl<'a> DecorateLint<'a, ()> for NonFmtPanicUnused {
    fn decorate_lint<'b>(
        self,
        diag: &'b mut rustc_errors::DiagnosticBuilder<'a, ()>,
    ) -> &'b mut rustc_errors::DiagnosticBuilder<'a, ()> {
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
        diag
    }

    fn msg(&self) -> rustc_errors::DiagnosticMessage {
        fluent::lint_non_fmt_panic_unused
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

// nonstandard_style.rs
#[derive(LintDiagnostic)]
#[diag(lint_non_camel_case_type)]
pub struct NonCamelCaseType<'a> {
    pub sort: &'a str,
    pub name: &'a str,
    #[subdiagnostic]
    pub sub: NonCamelCaseTypeSub,
}

#[derive(Subdiagnostic)]
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

impl AddToDiagnostic for NonSnakeCaseDiagSub {
    fn add_to_diagnostic_with<F>(self, diag: &mut rustc_errors::Diagnostic, _: F)
    where
        F: Fn(
            &mut rustc_errors::Diagnostic,
            rustc_errors::SubdiagnosticMessage,
        ) -> rustc_errors::SubdiagnosticMessage,
    {
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
        }
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

#[derive(Subdiagnostic)]
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

// noop_method_call.rs
#[derive(LintDiagnostic)]
#[diag(lint_noop_method_call)]
#[note]
pub struct NoopMethodCallDiag<'a> {
    pub method: Symbol,
    pub receiver_ty: Ty<'a>,
    #[label]
    pub label: Span,
}

// pass_by_value.rs
#[derive(LintDiagnostic)]
#[diag(lint_pass_by_value)]
pub struct PassByValueDiag {
    pub ty: String,
    #[suggestion(code = "{ty}", applicability = "maybe-incorrect")]
    pub suggestion: Span,
}

// redundant_semicolon.rs
#[derive(LintDiagnostic)]
#[diag(lint_redundant_semicolons)]
pub struct RedundantSemicolonsDiag {
    pub multiple: bool,
    #[suggestion(code = "", applicability = "maybe-incorrect")]
    pub suggestion: Span,
}

// traits.rs
pub struct DropTraitConstraintsDiag<'a> {
    pub predicate: Predicate<'a>,
    pub tcx: TyCtxt<'a>,
    pub def_id: DefId,
}

impl<'a> DecorateLint<'a, ()> for DropTraitConstraintsDiag<'_> {
    fn decorate_lint<'b>(
        self,
        diag: &'b mut rustc_errors::DiagnosticBuilder<'a, ()>,
    ) -> &'b mut rustc_errors::DiagnosticBuilder<'a, ()> {
        diag.set_arg("predicate", self.predicate);
        diag.set_arg("needs_drop", self.tcx.def_path_str(self.def_id))
    }

    fn msg(&self) -> rustc_errors::DiagnosticMessage {
        fluent::lint_drop_trait_constraints
    }
}

pub struct DropGlue<'a> {
    pub tcx: TyCtxt<'a>,
    pub def_id: DefId,
}

impl<'a> DecorateLint<'a, ()> for DropGlue<'_> {
    fn decorate_lint<'b>(
        self,
        diag: &'b mut rustc_errors::DiagnosticBuilder<'a, ()>,
    ) -> &'b mut rustc_errors::DiagnosticBuilder<'a, ()> {
        diag.set_arg("needs_drop", self.tcx.def_path_str(self.def_id))
    }

    fn msg(&self) -> rustc_errors::DiagnosticMessage {
        fluent::lint_drop_glue
    }
}

// types.rs
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

impl AddToDiagnostic for OverflowingBinHexSign {
    fn add_to_diagnostic_with<F>(self, diag: &mut rustc_errors::Diagnostic, _: F)
    where
        F: Fn(
            &mut rustc_errors::Diagnostic,
            rustc_errors::SubdiagnosticMessage,
        ) -> rustc_errors::SubdiagnosticMessage,
    {
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

#[derive(Subdiagnostic)]
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
impl<'a> DecorateLint<'a, ()> for OverflowingInt<'_> {
    fn decorate_lint<'b>(
        self,
        diag: &'b mut rustc_errors::DiagnosticBuilder<'a, ()>,
    ) -> &'b mut rustc_errors::DiagnosticBuilder<'a, ()> {
        diag.set_arg("ty", self.ty);
        diag.set_arg("lit", self.lit);
        diag.set_arg("min", self.min);
        diag.set_arg("max", self.max);
        diag.note(fluent::note);
        if let Some(suggestion_ty) = self.suggestion_ty {
            diag.set_arg("suggestion_ty", suggestion_ty);
            diag.help(fluent::help);
        }
        diag
    }

    fn msg(&self) -> rustc_errors::DiagnosticMessage {
        fluent::lint_overflowing_int
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

pub struct ImproperCTypes<'a> {
    pub ty: Ty<'a>,
    pub desc: &'a str,
    pub label: Span,
    pub help: Option<DiagnosticMessage>,
    pub note: DiagnosticMessage,
    pub span_note: Option<Span>,
}

impl<'a> DecorateLint<'a, ()> for ImproperCTypes<'_> {
    fn decorate_lint<'b>(
        self,
        diag: &'b mut rustc_errors::DiagnosticBuilder<'a, ()>,
    ) -> &'b mut rustc_errors::DiagnosticBuilder<'a, ()> {
        diag.set_arg("ty", self.ty);
        diag.set_arg("desc", self.desc);
        diag.span_label(self.label, fluent::label);
        if let Some(help) = self.help {
            diag.help(help);
        }
        diag.note(self.note);
        if let Some(note) = self.span_note {
            diag.span_note(note, fluent::note);
        }
        diag
    }

    fn msg(&self) -> rustc_errors::DiagnosticMessage {
        fluent::lint_improper_ctypes
    }
}

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

// unused.rs
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
impl<'a> DecorateLint<'a, ()> for UnusedDef<'_, '_> {
    fn decorate_lint<'b>(
        self,
        diag: &'b mut rustc_errors::DiagnosticBuilder<'a, ()>,
    ) -> &'b mut rustc_errors::DiagnosticBuilder<'a, ()> {
        diag.set_arg("pre", self.pre);
        diag.set_arg("post", self.post);
        diag.set_arg("def", self.cx.tcx.def_path_str(self.def_id));
        // check for #[must_use = "..."]
        if let Some(note) = self.note {
            diag.note(note.as_str());
        }
        diag
    }

    fn msg(&self) -> rustc_errors::DiagnosticMessage {
        fluent::lint_unused_def
    }
}

#[derive(LintDiagnostic)]
#[diag(lint_path_statement_drop)]
pub struct PathStatementDrop {
    #[subdiagnostic]
    pub sub: PathStatementDropSub,
}

#[derive(Subdiagnostic)]
pub enum PathStatementDropSub {
    #[suggestion(suggestion, code = "drop({snippet});", applicability = "machine-applicable")]
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
