use std::borrow::Cow;

use rustc_ast::ast;
use rustc_ast::token::{NonterminalKind, Token};
use rustc_ast::tokenstream::TokenTree;
use rustc_errors::codes::*;
use rustc_errors::{DiagArgValue, IntoDiagArg, MultiSpan};
use rustc_macros::{Diagnostic, Subdiagnostic};
use rustc_session::errors::FeatureGateSubdiagnostic;
use rustc_session::Limit;
use rustc_span::symbol::{Ident, MacroRulesNormalizedIdent};
use rustc_span::{Span, Symbol};

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
#[diag(expand_meta_var_expr_unrecognized_var)]
pub(crate) struct MetaVarExprUnrecognizedVar {
    #[primary_span]
    pub span: Span,
    pub key: MacroRulesNormalizedIdent,
}

#[derive(Diagnostic)]
#[diag(expand_var_still_repeating)]
pub(crate) struct VarStillRepeating {
    #[primary_span]
    pub span: Span,
    pub ident: MacroRulesNormalizedIdent,
}

#[derive(Diagnostic)]
#[diag(expand_meta_var_dif_seq_matchers)]
pub(crate) struct MetaVarsDifSeqMatchers {
    #[primary_span]
    pub span: Span,
    pub var1_id: String,
    pub var1_len: usize,
    pub var2_id: String,
    pub var2_len: usize,
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
#[diag(expand_attr_no_arguments)]
pub(crate) struct AttrNoArguments {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(expand_not_a_meta_item)]
pub(crate) struct NotAMetaItem {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(expand_only_one_word)]
pub(crate) struct OnlyOneWord {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(expand_cannot_be_name_of_macro)]
pub(crate) struct CannotBeNameOfMacro<'a> {
    #[primary_span]
    pub span: Span,
    pub trait_ident: Ident,
    pub macro_type: &'a str,
}

#[derive(Diagnostic)]
#[diag(expand_arg_not_attributes)]
pub(crate) struct ArgumentNotAttributes {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(expand_attributes_wrong_form)]
pub(crate) struct AttributesWrongForm {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(expand_attribute_meta_item)]
pub(crate) struct AttributeMetaItem {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(expand_attribute_single_word)]
pub(crate) struct AttributeSingleWord {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(expand_helper_attribute_name_invalid)]
pub(crate) struct HelperAttributeNameInvalid {
    #[primary_span]
    pub span: Span,
    pub name: Ident,
}

#[derive(Diagnostic)]
#[diag(expand_feature_removed, code = E0557)]
pub(crate) struct FeatureRemoved<'a> {
    #[primary_span]
    #[label]
    pub span: Span,
    #[subdiagnostic]
    pub reason: Option<FeatureRemovedReason<'a>>,
}

#[derive(Subdiagnostic)]
#[note(expand_reason)]
pub(crate) struct FeatureRemovedReason<'a> {
    // FIXME: make this translatable
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
pub(crate) struct RecursionLimitReached<'a> {
    #[primary_span]
    pub span: Span,
    pub descr: String,
    pub suggested_limit: Limit,
    pub crate_name: &'a str,
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
    pub token: Cow<'a, str>,
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
    // FIXME: make this translatable
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
pub(crate) struct ModuleInBlock {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub name: Option<ModuleInBlockName>,
}

#[derive(Subdiagnostic)]
#[note(expand_note)]
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

#[derive(Subdiagnostic)]
pub(crate) enum InvalidFragmentSpecifierValidNames {
    #[help(expand_valid_fragment_names_2021)]
    Edition2021,
    #[help(expand_valid_fragment_names_other)]
    Other,
}

#[derive(Diagnostic)]
#[diag(expand_missing_fragment_specifier)]
#[note]
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
    #[subdiagnostic]
    pub valid: InvalidFragmentSpecifierValidNames,
}

#[derive(Diagnostic)]
#[diag(expand_invalid_fragment_specifier)]
pub(crate) struct InvalidFragmentSpecifier {
    #[primary_span]
    pub span: Span,
    #[help(expand_help_expr_2021)]
    pub help_expr_2021: bool,
    #[subdiagnostic]
    pub help_valid_names: InvalidFragmentSpecifierValidNames,
    pub fragment: Ident,
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

pub(crate) enum StatementOrExpression {
    Statement,
    Expression,
}

impl IntoDiagArg for StatementOrExpression {
    fn into_diag_arg(self) -> rustc_errors::DiagArgValue {
        let s = match self {
            StatementOrExpression::Statement => "statement",
            StatementOrExpression::Expression => "expression",
        };

        rustc_errors::DiagArgValue::Str(s.into())
    }
}

#[derive(Diagnostic)]
#[diag(expand_custom_attribute_cannot_be_applied, code = E0658)]
pub(crate) struct CustomAttributesForbidden {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub subdiag: FeatureGateSubdiagnostic,
    pub kind: StatementOrExpression,
}

#[derive(Diagnostic)]
#[diag(expand_non_inline_modules_in_proc_macro_input_are_unstable, code = E0658)]
pub(crate) struct NonInlineModuleInProcMacroUnstable {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub subdiag: FeatureGateSubdiagnostic,
}

#[derive(Subdiagnostic)]
#[label(expand_label_error_while_parsing_argument)]
pub(crate) struct NoteParseErrorInMacroArgument {
    #[primary_span]
    pub span: Span,
    pub kind: NonterminalKind,
}

#[derive(Diagnostic)]
#[diag(expand_meta_var_expr_needs_parens)]
pub(crate) struct MetaVarExprParamNeedsParens {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(expand_unsupported_concat_elem)]
pub(crate) struct UnsupportedConcatElem {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(expand_concat_raw_ident)]
pub(crate) struct ConcatRawIdent {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(expand_expected_comma)]
pub(crate) struct ExpectedComma {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(expand_concat_too_few_args)]
pub(crate) struct ConcatTooFewArgs {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(expand_unrecognized_meta_var_expr)]
#[help]
pub(crate) struct UnrecognizedMetaVarExpr {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(expand_count_with_comma_no_index)]
pub(crate) struct CountWithCommaNoIndex {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(expand_nested_meta_var_expr_without_dollar)]
pub(crate) struct NestedMetaVarExprWithoutDollar {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(expand_meta_var_expr_depth_not_literal)]
pub(crate) struct MetaVarExprDepthNotLiteral {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(expand_meta_var_expr_depth_suffixed)]
pub(crate) struct MetaVarExprDepthSuffixed {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(expand_meta_var_expr_unexpected_token)]
pub(crate) struct MetaVarExprUnexpectedToken {
    #[primary_span]
    #[note]
    pub span: Span,
    pub tt: TokenTree,
}

#[derive(Diagnostic)]
#[diag(expand_meta_var_expr_expected_identifier)]
pub(crate) struct MetaVarExprExpectedIdentifier {
    #[primary_span]
    #[suggestion(code = "", applicability = "maybe-incorrect")]
    pub span: Span,
    pub found: Token,
}

#[derive(Diagnostic)]
#[diag(expand_expected_identifier)]
pub(crate) struct ExpectedIdentifier {
    #[primary_span]
    pub span: Span,
    pub found: Token,
}

#[derive(Diagnostic)]
#[diag(expand_question_mark_with_separator)]
pub(crate) struct QuestionMarkWithSeparator {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(expand_expected_repetition_operator)]
pub(crate) struct ExpectedRepetitionOperator {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(expand_dollar_or_metavar_in_lhs)]
pub(crate) struct DollarOrMetavarInLhs {
    #[primary_span]
    #[note]
    pub span: Span,
    pub token: Token,
}

#[derive(Diagnostic)]
#[diag(expand_meta_var_expr_out_of_bounds)]
pub(crate) struct MetaVarExprOutOfBounds {
    #[primary_span]
    pub span: Span,
    pub ty: String,
    pub max: usize,
}

#[derive(Diagnostic)]
#[diag(expand_concat_generated_invalid_ident)]
pub(crate) struct ConcatGeneratedInvalidIdent {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(expand_invalid_concat_arg_type)]
#[note]
pub(crate) struct InvalidConcatArgType {
    #[primary_span]
    pub span: Span,
}

#[derive(Subdiagnostic)]
pub(crate) enum MatchFailureLabel {
    #[label(expand_match_failure_missing_tokens)]
    MissingTokens(#[primary_span] Span),
    #[label(expand_match_failure_unexpected_token)]
    UnexpectedToken(#[primary_span] Span),
}

#[derive(Subdiagnostic)]
pub(crate) enum ExplainDocComment {
    #[label(expand_explain_doc_comment_inner)]
    Inner {
        #[primary_span]
        span: Span,
    },
    #[label(expand_explain_doc_comment_outer)]
    Outer {
        #[primary_span]
        span: Span,
    },
}

#[derive(Subdiagnostic)]
pub(crate) struct ParseFailureSubdiags {
    #[subdiagnostic]
    pub failure_label: MatchFailureLabel,
    #[subdiagnostic]
    pub doc_comment: Option<ExplainDocComment>,
}

#[derive(Diagnostic)]
pub(crate) enum ParseFailure {
    #[diag(expand_parse_failure_expected_token)]
    ExpectedToken {
        #[primary_span]
        span: Span,
        expected: Token,
        found: Token,
        #[subdiagnostic]
        subdiags: ParseFailureSubdiags,
    },
    #[diag(expand_parse_failure_unexpected_eof)]
    UnexpectedEof {
        #[primary_span]
        span: Span,
        #[subdiagnostic]
        subdiags: ParseFailureSubdiags,
    },
    #[diag(expand_parse_failure_unexpected_token)]
    UnexpectedToken {
        #[primary_span]
        span: Span,
        found: Token,
        #[subdiagnostic]
        subdiags: ParseFailureSubdiags,
    },
}

#[derive(Diagnostic)]
#[diag(expand_unknown_macro_transparency)]
pub(crate) struct UnknownMacroTransparency {
    #[primary_span]
    pub span: Span,
    pub value: Symbol,
}

#[derive(Diagnostic)]
#[diag(expand_multiple_transparency_attrs)]
pub(crate) struct MultipleTransparencyAttrs {
    #[primary_span]
    pub spans: MultiSpan,
}

#[derive(Diagnostic)]
#[diag(expand_unbalanced_delims_around_matcher)]
pub(crate) struct UnbalancedDelimsAroundMatcher {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(expand_doc_comments_ignored_in_matcher_position)]
pub(crate) struct DocCommentsIgnoredInMatcherPosition {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(expand_repetition_matches_empty_token_tree)]
pub(crate) struct RepetitionMatchesEmptyTokenTree {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(expand_macro_rhs_must_be_delimited)]
pub(crate) struct MacroRhsMustBeDelimited {
    #[primary_span]
    pub span: Span,
}

#[derive(Subdiagnostic)]
#[suggestion(expand_suggestion, code = "{suggestion}", applicability = "maybe-incorrect")]
pub(crate) struct InvalidFollowSuggestion {
    #[primary_span]
    pub span: Span,
    pub suggestion: String,
}

#[derive(Subdiagnostic)]
#[note(expand_note)]
pub(crate) struct InvalidFollowNote {
    pub num_possible: usize,
    pub possible: DiagArgValue,
}

#[derive(Diagnostic)]
#[diag(expand_invalid_follow)]
pub(crate) struct InvalidFollow {
    #[primary_span]
    #[label]
    pub span: Span,
    pub name: Ident,
    pub kind: NonterminalKind,
    pub next: String,
    pub only_option: bool,

    #[subdiagnostic]
    pub suggestion: Option<InvalidFollowSuggestion>,
    #[subdiagnostic]
    pub note_allowed: Option<InvalidFollowNote>,
}

// FIXME: unify this with MissingFragmentSpecifier
#[derive(Diagnostic)]
#[diag(expand_missing_fragment_specifier)]
pub(crate) struct MissingFragmentSpecifierThin {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(expand_multiple_successful_parses)]
pub(crate) struct MultipleSuccessfulParses {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(expand_duplicate_binding_name)]
pub(crate) struct DuplicateBindingName {
    #[primary_span]
    pub span: Span,
    pub bind: Ident,
}

#[derive(Diagnostic)]
#[diag(expand_multiple_parsing_options)]
pub(crate) struct MultipleParsingOptions {
    #[primary_span]
    pub span: Span,
    pub macro_name: Ident,
    pub n: usize,
    pub nts: DiagArgValue,
}
