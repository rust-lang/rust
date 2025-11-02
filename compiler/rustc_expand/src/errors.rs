use std::borrow::Cow;

use rustc_ast::ast;
use rustc_errors::codes::*;
use rustc_hir::limit::Limit;
use rustc_macros::{Diagnostic, LintDiagnostic, Subdiagnostic};
use rustc_span::{Ident, MacroRulesNormalizedIdent, Span, Symbol};

#[derive(LintDiagnostic)]
#[diag(expand_cfg_attr_no_attributes)]
pub(crate) struct CfgAttrNoAttributes;

#[derive(Diagnostic)]
#[diag(expand_expr_repeat_no_syntax_vars)]
pub(crate) struct NoSyntaxVarsExprRepeat {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(expand_must_repeat_once)]
pub(crate) struct MustRepeatOnce {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(expand_count_repetition_misplaced)]
pub(crate) struct CountRepetitionMisplaced {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(expand_metavar_still_repeating)]
pub(crate) struct MacroVarStillRepeating {
    #[primary_span]
    pub span: Span,
    pub ident: MacroRulesNormalizedIdent,
}

#[derive(LintDiagnostic)]
#[diag(expand_metavar_still_repeating)]
pub(crate) struct MetaVarStillRepeatingLint {
    #[label]
    pub label: Span,
    pub ident: MacroRulesNormalizedIdent,
}

#[derive(LintDiagnostic)]
#[diag(expand_metavariable_wrong_operator)]
pub(crate) struct MetaVariableWrongOperator {
    #[label(expand_binder_label)]
    pub binder: Span,
    #[label(expand_occurrence_label)]
    pub occurrence: Span,
}

#[derive(Diagnostic)]
#[diag(expand_meta_var_dif_seq_matchers)]
pub(crate) struct MetaVarsDifSeqMatchers {
    #[primary_span]
    pub span: Span,
    pub msg: String,
}

#[derive(LintDiagnostic)]
#[diag(expand_unknown_macro_variable)]
pub(crate) struct UnknownMacroVariable {
    pub name: MacroRulesNormalizedIdent,
}

#[derive(Diagnostic)]
#[diag(expand_resolve_relative_path)]
pub(crate) struct ResolveRelativePath {
    #[primary_span]
    pub span: Span,
    pub path: String,
}

#[derive(Diagnostic)]
#[diag(expand_collapse_debuginfo_illegal)]
pub(crate) struct CollapseMacroDebuginfoIllegal {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(expand_macro_const_stability)]
pub(crate) struct MacroConstStability {
    #[primary_span]
    #[label]
    pub span: Span,
    #[label(expand_label2)]
    pub head_span: Span,
}

#[derive(Diagnostic)]
#[diag(expand_macro_body_stability)]
pub(crate) struct MacroBodyStability {
    #[primary_span]
    #[label]
    pub span: Span,
    #[label(expand_label2)]
    pub head_span: Span,
}

#[derive(Diagnostic)]
#[diag(expand_feature_removed, code = E0557)]
#[note]
pub(crate) struct FeatureRemoved<'a> {
    #[primary_span]
    #[label]
    pub span: Span,
    #[subdiagnostic]
    pub reason: Option<FeatureRemovedReason<'a>>,
    pub removed_rustc_version: &'a str,
    pub pull_note: String,
}

#[derive(Subdiagnostic)]
#[note(expand_reason)]
pub(crate) struct FeatureRemovedReason<'a> {
    pub reason: &'a str,
}

#[derive(Diagnostic)]
#[diag(expand_feature_not_allowed, code = E0725)]
pub(crate) struct FeatureNotAllowed {
    #[primary_span]
    pub span: Span,
    pub name: Symbol,
}

#[derive(Diagnostic)]
#[diag(expand_recursion_limit_reached)]
#[help]
pub(crate) struct RecursionLimitReached {
    #[primary_span]
    pub span: Span,
    pub descr: String,
    pub suggested_limit: Limit,
    pub crate_name: Symbol,
}

#[derive(Diagnostic)]
#[diag(expand_malformed_feature_attribute, code = E0556)]
pub(crate) struct MalformedFeatureAttribute {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub help: MalformedFeatureAttributeHelp,
}

#[derive(Subdiagnostic)]
pub(crate) enum MalformedFeatureAttributeHelp {
    #[label(expand_expected)]
    Label {
        #[primary_span]
        span: Span,
    },
    #[suggestion(expand_expected, code = "{suggestion}", applicability = "maybe-incorrect")]
    Suggestion {
        #[primary_span]
        span: Span,
        suggestion: Symbol,
    },
}

#[derive(Diagnostic)]
#[diag(expand_remove_expr_not_supported)]
pub(crate) struct RemoveExprNotSupported {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
pub(crate) enum InvalidCfg {
    #[diag(expand_invalid_cfg_no_parens)]
    NotFollowedByParens {
        #[primary_span]
        #[suggestion(
            expand_invalid_cfg_expected_syntax,
            code = "cfg(/* predicate */)",
            applicability = "has-placeholders"
        )]
        span: Span,
    },
    #[diag(expand_invalid_cfg_no_predicate)]
    NoPredicate {
        #[primary_span]
        #[suggestion(
            expand_invalid_cfg_expected_syntax,
            code = "cfg(/* predicate */)",
            applicability = "has-placeholders"
        )]
        span: Span,
    },
    #[diag(expand_invalid_cfg_multiple_predicates)]
    MultiplePredicates {
        #[primary_span]
        span: Span,
    },
    #[diag(expand_invalid_cfg_predicate_literal)]
    PredicateLiteral {
        #[primary_span]
        span: Span,
    },
}

#[derive(Diagnostic)]
#[diag(expand_wrong_fragment_kind)]
pub(crate) struct WrongFragmentKind<'a> {
    #[primary_span]
    pub span: Span,
    pub kind: &'a str,
    pub name: &'a ast::Path,
}

#[derive(Diagnostic)]
#[diag(expand_unsupported_key_value)]
pub(crate) struct UnsupportedKeyValue {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(expand_incomplete_parse)]
#[note]
pub(crate) struct IncompleteParse<'a> {
    #[primary_span]
    pub span: Span,
    pub descr: String,
    #[label]
    pub label_span: Span,
    pub macro_path: &'a ast::Path,
    pub kind_name: &'a str,
    #[note(expand_macro_expands_to_match_arm)]
    pub expands_to_match_arm: bool,

    #[suggestion(
        expand_suggestion_add_semi,
        style = "verbose",
        code = ";",
        applicability = "maybe-incorrect"
    )]
    pub add_semicolon: Option<Span>,
}

#[derive(Diagnostic)]
#[diag(expand_remove_node_not_supported)]
pub(crate) struct RemoveNodeNotSupported {
    #[primary_span]
    pub span: Span,
    pub descr: &'static str,
}

#[derive(Diagnostic)]
#[diag(expand_module_circular)]
pub(crate) struct ModuleCircular {
    #[primary_span]
    pub span: Span,
    pub modules: String,
}

#[derive(Diagnostic)]
#[diag(expand_module_in_block)]
#[note]
pub(crate) struct ModuleInBlock {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub name: Option<ModuleInBlockName>,
}

#[derive(Subdiagnostic)]
#[help(expand_help)]
pub(crate) struct ModuleInBlockName {
    #[primary_span]
    pub span: Span,
    pub name: Ident,
}

#[derive(Diagnostic)]
#[diag(expand_module_file_not_found, code = E0583)]
#[help]
#[note]
pub(crate) struct ModuleFileNotFound {
    #[primary_span]
    pub span: Span,
    pub name: Ident,
    pub default_path: String,
    pub secondary_path: String,
}

#[derive(Diagnostic)]
#[diag(expand_module_multiple_candidates, code = E0761)]
#[help]
pub(crate) struct ModuleMultipleCandidates {
    #[primary_span]
    pub span: Span,
    pub name: Ident,
    pub default_path: String,
    pub secondary_path: String,
}

#[derive(Diagnostic)]
#[diag(expand_trace_macro)]
pub(crate) struct TraceMacro {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(expand_proc_macro_panicked)]
pub(crate) struct ProcMacroPanicked {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub message: Option<ProcMacroPanickedHelp>,
}

#[derive(Subdiagnostic)]
#[help(expand_help)]
pub(crate) struct ProcMacroPanickedHelp {
    pub message: String,
}

#[derive(Diagnostic)]
#[diag(expand_proc_macro_derive_panicked)]
pub(crate) struct ProcMacroDerivePanicked {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub message: Option<ProcMacroDerivePanickedHelp>,
}

#[derive(Subdiagnostic)]
#[help(expand_help)]
pub(crate) struct ProcMacroDerivePanickedHelp {
    pub message: String,
}

#[derive(Diagnostic)]
#[diag(expand_custom_attribute_panicked)]
pub(crate) struct CustomAttributePanicked {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub message: Option<CustomAttributePanickedHelp>,
}

#[derive(Subdiagnostic)]
#[help(expand_help)]
pub(crate) struct CustomAttributePanickedHelp {
    pub message: String,
}

#[derive(Diagnostic)]
#[diag(expand_proc_macro_derive_tokens)]
pub(crate) struct ProcMacroDeriveTokens {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(expand_duplicate_matcher_binding)]
pub(crate) struct DuplicateMatcherBinding {
    #[primary_span]
    #[label]
    pub span: Span,
    #[label(expand_label2)]
    pub prev: Span,
}

#[derive(LintDiagnostic)]
#[diag(expand_duplicate_matcher_binding)]
pub(crate) struct DuplicateMatcherBindingLint {
    #[label]
    pub span: Span,
    #[label(expand_label2)]
    pub prev: Span,
}

#[derive(Diagnostic)]
#[diag(expand_missing_fragment_specifier)]
#[note]
#[help(expand_valid)]
pub(crate) struct MissingFragmentSpecifier {
    #[primary_span]
    pub span: Span,
    #[suggestion(
        expand_suggestion_add_fragspec,
        style = "verbose",
        code = ":spec",
        applicability = "maybe-incorrect"
    )]
    pub add_span: Span,
    pub valid: &'static str,
}

#[derive(Diagnostic)]
#[diag(expand_invalid_fragment_specifier)]
#[help]
pub(crate) struct InvalidFragmentSpecifier {
    #[primary_span]
    pub span: Span,
    pub fragment: Ident,
    pub help: &'static str,
}

#[derive(Diagnostic)]
#[diag(expand_expected_paren_or_brace)]
pub(crate) struct ExpectedParenOrBrace<'a> {
    #[primary_span]
    pub span: Span,
    pub token: Cow<'a, str>,
}

#[derive(Diagnostic)]
#[diag(expand_empty_delegation_mac)]
pub(crate) struct EmptyDelegationMac {
    #[primary_span]
    pub span: Span,
    pub kind: String,
}

#[derive(Diagnostic)]
#[diag(expand_glob_delegation_outside_impls)]
pub(crate) struct GlobDelegationOutsideImpls {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(expand_crate_name_in_cfg_attr)]
pub(crate) struct CrateNameInCfgAttr {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(expand_crate_type_in_cfg_attr)]
pub(crate) struct CrateTypeInCfgAttr {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(expand_glob_delegation_traitless_qpath)]
pub(crate) struct GlobDelegationTraitlessQpath {
    #[primary_span]
    pub span: Span,
}

// This used to be the `proc_macro_back_compat` lint (#83125). It was later
// turned into a hard error.
#[derive(Diagnostic)]
#[diag(expand_proc_macro_back_compat)]
#[note]
pub(crate) struct ProcMacroBackCompat {
    pub crate_name: String,
    pub fixed_version: String,
}

pub(crate) use metavar_exprs::*;
mod metavar_exprs {
    use super::*;

    #[derive(Diagnostic, Default)]
    #[diag(expand_mve_extra_tokens)]
    pub(crate) struct MveExtraTokens {
        #[primary_span]
        #[suggestion(code = "", applicability = "machine-applicable")]
        pub span: Span,
        #[label]
        pub ident_span: Span,
        pub extra_count: usize,

        // The rest is only used for specific diagnostics and can be default if neither
        // `note` is `Some`.
        #[note(expand_exact)]
        pub exact_args_note: Option<()>,
        #[note(expand_range)]
        pub range_args_note: Option<()>,
        pub min_or_exact_args: usize,
        pub max_args: usize,
        pub name: String,
    }

    #[derive(Diagnostic)]
    #[note]
    #[diag(expand_mve_missing_paren)]
    pub(crate) struct MveMissingParen {
        #[primary_span]
        #[label]
        pub ident_span: Span,
        #[label(expand_unexpected)]
        pub unexpected_span: Option<Span>,
        #[suggestion(code = "( /* ... */ )", applicability = "has-placeholders")]
        pub insert_span: Option<Span>,
    }

    #[derive(Diagnostic)]
    #[note]
    #[diag(expand_mve_unrecognized_expr)]
    pub(crate) struct MveUnrecognizedExpr {
        #[primary_span]
        #[label]
        pub span: Span,
        pub valid_expr_list: &'static str,
    }

    #[derive(Diagnostic)]
    #[diag(expand_mve_unrecognized_var)]
    pub(crate) struct MveUnrecognizedVar {
        #[primary_span]
        pub span: Span,
        pub key: MacroRulesNormalizedIdent,
    }
}

#[derive(Diagnostic)]
#[diag(expand_macro_args_bad_delim)]
pub(crate) struct MacroArgsBadDelim {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub sugg: MacroArgsBadDelimSugg,
    pub rule_kw: Symbol,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(expand_macro_args_bad_delim_sugg, applicability = "machine-applicable")]
pub(crate) struct MacroArgsBadDelimSugg {
    #[suggestion_part(code = "(")]
    pub open: Span,
    #[suggestion_part(code = ")")]
    pub close: Span,
}

#[derive(LintDiagnostic)]
#[diag(expand_macro_call_unused_doc_comment)]
#[help]
pub(crate) struct MacroCallUnusedDocComment {
    #[label]
    pub span: Span,
}

#[derive(LintDiagnostic)]
#[diag(expand_or_patterns_back_compat)]
pub(crate) struct OrPatternsBackCompat {
    #[suggestion(code = "{suggestion}", applicability = "machine-applicable")]
    pub span: Span,
    pub suggestion: String,
}

#[derive(LintDiagnostic)]
#[diag(expand_trailing_semi_macro)]
pub(crate) struct TrailingMacro {
    #[note(expand_note1)]
    #[note(expand_note2)]
    pub is_trailing: bool,
    pub name: Ident,
}

#[derive(LintDiagnostic)]
#[diag(expand_unused_builtin_attribute)]
pub(crate) struct UnusedBuiltinAttribute {
    #[note]
    pub invoc_span: Span,
    pub attr_name: Symbol,
    pub macro_name: String,
    #[suggestion(code = "", applicability = "machine-applicable", style = "tool-only")]
    pub attr_span: Span,
}
