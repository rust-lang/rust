use rustc_ast::ast;
use rustc_macros::Diagnostic;
use rustc_session::Limit;
use rustc_span::edition::Edition;
use rustc_span::symbol::{Ident, MacroRulesNormalizedIdent};
use rustc_span::{Span, Symbol};
use std::borrow::Cow;

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
    pub msg: String,
}

#[derive(Diagnostic)]
#[diag(expand_resolve_relative_path)]
pub(crate) struct ResolveRelativePath {
    #[primary_span]
    pub span: Span,
    pub path: String,
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
#[diag(expand_expected_comma_in_list)]
pub(crate) struct ExpectedCommaInList {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(expand_only_one_argument)]
pub(crate) struct OnlyOneArgument<'a> {
    #[primary_span]
    pub span: Span,
    pub name: &'a str,
}

#[derive(Diagnostic)]
#[diag(expand_takes_no_arguments)]
pub(crate) struct TakesNoArguments<'a> {
    #[primary_span]
    pub span: Span,
    pub name: &'a str,
}

#[derive(Diagnostic)]
#[diag(expand_feature_included_in_edition, code = "E0705")]
pub(crate) struct FeatureIncludedInEdition {
    #[primary_span]
    pub span: Span,
    pub feature: Symbol,
    pub edition: Edition,
}

#[derive(Diagnostic)]
#[diag(expand_feature_removed, code = "E0557")]
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
    pub reason: &'a str,
}

#[derive(Diagnostic)]
#[diag(expand_feature_not_allowed, code = "E0725")]
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
#[diag(expand_malformed_feature_attribute, code = "E0556")]
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
#[diag(expand_module_file_not_found, code = "E0583")]
#[help]
pub(crate) struct ModuleFileNotFound {
    #[primary_span]
    pub span: Span,
    pub name: Ident,
    pub default_path: String,
    pub secondary_path: String,
}

#[derive(Diagnostic)]
#[diag(expand_module_multiple_candidates, code = "E0761")]
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
pub struct TraceMacro {
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
#[diag(expand_proc_macro_derive_tokens)]
pub struct ProcMacroDeriveTokens {
    #[primary_span]
    pub span: Span,
}
