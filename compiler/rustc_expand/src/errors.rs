use std::borrow::Cow;

use rustc_ast::ast;
use rustc_errors::codes::*;
use rustc_hir::limit::Limit;
use rustc_macros::{Diagnostic, LintDiagnostic, Subdiagnostic};
use rustc_span::{Ident, MacroRulesNormalizedIdent, Span, Symbol};

#[derive(LintDiagnostic)]
#[diag("`#[cfg_attr]` does not expand to any attributes")]
pub(crate) struct CfgAttrNoAttributes;

#[derive(Diagnostic)]
#[diag(
    "attempted to repeat an expression containing no syntax variables matched as repeating at this depth"
)]
pub(crate) struct NoSyntaxVarsExprRepeat {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub typo_repeatable: Option<VarTypoSuggestionRepeatable>,
    #[subdiagnostic]
    pub typo_unrepeatable: Option<VarTypoSuggestionUnrepeatable>,
    #[subdiagnostic]
    pub typo_unrepeatable_label: Option<VarTypoSuggestionUnrepeatableLabel>,
    #[subdiagnostic]
    pub var_no_typo: Option<VarNoTypo>,
    #[subdiagnostic]
    pub no_repeatable_var: Option<NoRepeatableVar>,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    "there's a macro metavariable with a similar name",
    applicability = "maybe-incorrect",
    style = "verbose"
)]
pub(crate) struct VarTypoSuggestionRepeatable {
    #[suggestion_part(code = "{name}")]
    pub span: Span,
    pub name: Symbol,
}

#[derive(Subdiagnostic)]
#[label("argument not found")]
pub(crate) struct VarTypoSuggestionUnrepeatable {
    #[primary_span]
    pub span: Span,
}

#[derive(Subdiagnostic)]
#[label("this similarly named macro metavariable is unrepeatable")]
pub(crate) struct VarTypoSuggestionUnrepeatableLabel {
    #[primary_span]
    pub span: Span,
}

#[derive(Subdiagnostic)]
#[label("expected a repeatable metavariable: {$msg}")]
pub(crate) struct VarNoTypo {
    #[primary_span]
    pub span: Span,
    pub msg: String,
}

#[derive(Subdiagnostic)]
#[label(
    "this macro metavariable is not repeatable and there are no other repeatable metavariables"
)]
pub(crate) struct NoRepeatableVar {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("this must repeat at least once")]
pub(crate) struct MustRepeatOnce {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("`count` can not be placed inside the innermost repetition")]
pub(crate) struct CountRepetitionMisplaced {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("variable `{$ident}` is still repeating at this depth")]
pub(crate) struct MacroVarStillRepeating {
    #[primary_span]
    pub span: Span,
    pub ident: MacroRulesNormalizedIdent,
}

#[derive(LintDiagnostic)]
#[diag("variable `{$ident}` is still repeating at this depth")]
pub(crate) struct MetaVarStillRepeatingLint {
    #[label("expected repetition")]
    pub label: Span,
    pub ident: MacroRulesNormalizedIdent,
}

#[derive(LintDiagnostic)]
#[diag("meta-variable repeats with different Kleene operator")]
pub(crate) struct MetaVariableWrongOperator {
    #[label("expected repetition")]
    pub binder: Span,
    #[label("conflicting repetition")]
    pub occurrence: Span,
}

#[derive(Diagnostic)]
#[diag("{$msg}")]
pub(crate) struct MetaVarsDifSeqMatchers {
    #[primary_span]
    pub span: Span,
    pub msg: String,
}

#[derive(LintDiagnostic)]
#[diag("unknown macro variable `{$name}`")]
pub(crate) struct UnknownMacroVariable {
    pub name: MacroRulesNormalizedIdent,
}

#[derive(Diagnostic)]
#[diag("cannot resolve relative path in non-file source `{$path}`")]
pub(crate) struct ResolveRelativePath {
    #[primary_span]
    pub span: Span,
    pub path: String,
}

#[derive(Diagnostic)]
#[diag("macros cannot have body stability attributes")]
pub(crate) struct MacroBodyStability {
    #[primary_span]
    #[label("invalid body stability attribute")]
    pub span: Span,
    #[label("body stability attribute affects this macro")]
    pub head_span: Span,
}

#[derive(Diagnostic)]
#[diag("feature has been removed", code = E0557)]
#[note("removed in {$removed_rustc_version}{$pull_note}")]
pub(crate) struct FeatureRemoved<'a> {
    #[primary_span]
    #[label("feature has been removed")]
    pub span: Span,
    #[subdiagnostic]
    pub reason: Option<FeatureRemovedReason<'a>>,
    pub removed_rustc_version: &'a str,
    pub pull_note: String,
}

#[derive(Subdiagnostic)]
#[note("{$reason}")]
pub(crate) struct FeatureRemovedReason<'a> {
    pub reason: &'a str,
}

#[derive(Diagnostic)]
#[diag("the feature `{$name}` is not in the list of allowed features", code = E0725)]
pub(crate) struct FeatureNotAllowed {
    #[primary_span]
    pub span: Span,
    pub name: Symbol,
}

#[derive(Diagnostic)]
#[diag("recursion limit reached while expanding `{$descr}`")]
#[help(
    "consider increasing the recursion limit by adding a `#![recursion_limit = \"{$suggested_limit}\"]` attribute to your crate (`{$crate_name}`)"
)]
pub(crate) struct RecursionLimitReached {
    #[primary_span]
    pub span: Span,
    pub descr: String,
    pub suggested_limit: Limit,
    pub crate_name: Symbol,
}

#[derive(Diagnostic)]
#[diag("malformed `feature` attribute input", code = E0556)]
pub(crate) struct MalformedFeatureAttribute {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub help: MalformedFeatureAttributeHelp,
}

#[derive(Subdiagnostic)]
pub(crate) enum MalformedFeatureAttributeHelp {
    #[label("expected just one word")]
    Label {
        #[primary_span]
        span: Span,
    },
    #[suggestion(
        "expected just one word",
        code = "{suggestion}",
        applicability = "maybe-incorrect"
    )]
    Suggestion {
        #[primary_span]
        span: Span,
        suggestion: Symbol,
    },
}

#[derive(Diagnostic)]
#[diag("removing an expression is not supported in this position")]
pub(crate) struct RemoveExprNotSupported {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
pub(crate) enum InvalidCfg {
    #[diag("`cfg` is not followed by parentheses")]
    NotFollowedByParens {
        #[primary_span]
        #[suggestion(
            "expected syntax is",
            code = "cfg(/* predicate */)",
            applicability = "has-placeholders"
        )]
        span: Span,
    },
    #[diag("`cfg` predicate is not specified")]
    NoPredicate {
        #[primary_span]
        #[suggestion(
            "expected syntax is",
            code = "cfg(/* predicate */)",
            applicability = "has-placeholders"
        )]
        span: Span,
    },
    #[diag("multiple `cfg` predicates are specified")]
    MultiplePredicates {
        #[primary_span]
        span: Span,
    },
    #[diag("`cfg` predicate key cannot be a literal")]
    PredicateLiteral {
        #[primary_span]
        span: Span,
    },
}

#[derive(Diagnostic)]
#[diag("non-{$kind} macro in {$kind} position: {$name}")]
pub(crate) struct WrongFragmentKind<'a> {
    #[primary_span]
    pub span: Span,
    pub kind: &'a str,
    pub name: &'a ast::Path,
}

#[derive(Diagnostic)]
#[diag("key-value macro attributes are not supported")]
pub(crate) struct UnsupportedKeyValue {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("macro expansion ignores {$descr} and any tokens following")]
#[note("the usage of `{$macro_path}!` is likely invalid in {$kind_name} context")]
pub(crate) struct IncompleteParse<'a> {
    #[primary_span]
    pub span: Span,
    pub descr: String,
    #[label("caused by the macro expansion here")]
    pub label_span: Span,
    pub macro_path: &'a ast::Path,
    pub kind_name: &'a str,
    #[note("macros cannot expand to match arms")]
    pub expands_to_match_arm: bool,

    #[suggestion(
        "you might be missing a semicolon here",
        style = "verbose",
        code = ";",
        applicability = "maybe-incorrect"
    )]
    pub add_semicolon: Option<Span>,
}

#[derive(Diagnostic)]
#[diag("removing {$descr} is not supported in this position")]
pub(crate) struct RemoveNodeNotSupported {
    #[primary_span]
    pub span: Span,
    pub descr: &'static str,
}

#[derive(Diagnostic)]
#[diag("circular modules: {$modules}")]
pub(crate) struct ModuleCircular {
    #[primary_span]
    pub span: Span,
    pub modules: String,
}

#[derive(Diagnostic)]
#[diag("cannot declare a file module inside a block unless it has a path attribute")]
#[note("file modules are usually placed outside of blocks, at the top level of the file")]
pub(crate) struct ModuleInBlock {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub name: Option<ModuleInBlockName>,
}

#[derive(Subdiagnostic)]
#[help("maybe `use` the module `{$name}` instead of redeclaring it")]
pub(crate) struct ModuleInBlockName {
    #[primary_span]
    pub span: Span,
    pub name: Ident,
}

#[derive(Diagnostic)]
#[diag("file not found for module `{$name}`", code = E0583)]
#[help("to create the module `{$name}`, create file \"{$default_path}\" or \"{$secondary_path}\"")]
#[note(
    "if there is a `mod {$name}` elsewhere in the crate already, import it with `use crate::...` instead"
)]
pub(crate) struct ModuleFileNotFound {
    #[primary_span]
    pub span: Span,
    pub name: Ident,
    pub default_path: String,
    pub secondary_path: String,
}

#[derive(Diagnostic)]
#[diag("file for module `{$name}` found at both \"{$default_path}\" and \"{$secondary_path}\"", code = E0761)]
#[help("delete or rename one of them to remove the ambiguity")]
pub(crate) struct ModuleMultipleCandidates {
    #[primary_span]
    pub span: Span,
    pub name: Ident,
    pub default_path: String,
    pub secondary_path: String,
}

#[derive(Diagnostic)]
#[diag("trace_macro")]
pub(crate) struct TraceMacro {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("proc macro panicked")]
pub(crate) struct ProcMacroPanicked {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub message: Option<ProcMacroPanickedHelp>,
}

#[derive(Subdiagnostic)]
#[help("message: {$message}")]
pub(crate) struct ProcMacroPanickedHelp {
    pub message: String,
}

#[derive(Diagnostic)]
#[diag("proc-macro derive panicked")]
pub(crate) struct ProcMacroDerivePanicked {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub message: Option<ProcMacroDerivePanickedHelp>,
}

#[derive(Subdiagnostic)]
#[help("message: {$message}")]
pub(crate) struct ProcMacroDerivePanickedHelp {
    pub message: String,
}

#[derive(Diagnostic)]
#[diag("custom attribute panicked")]
pub(crate) struct CustomAttributePanicked {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub message: Option<CustomAttributePanickedHelp>,
}

#[derive(Subdiagnostic)]
#[help("message: {$message}")]
pub(crate) struct CustomAttributePanickedHelp {
    pub message: String,
}

#[derive(Diagnostic)]
#[diag("proc-macro derive produced unparsable tokens")]
pub(crate) struct ProcMacroDeriveTokens {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("duplicate matcher binding")]
pub(crate) struct DuplicateMatcherBinding {
    #[primary_span]
    #[label("duplicate binding")]
    pub span: Span,
    #[label("previous binding")]
    pub prev: Span,
}

#[derive(LintDiagnostic)]
#[diag("duplicate matcher binding")]
pub(crate) struct DuplicateMatcherBindingLint {
    #[label("duplicate binding")]
    pub span: Span,
    #[label("previous binding")]
    pub prev: Span,
}

#[derive(Diagnostic)]
#[diag("missing fragment specifier")]
#[note("fragment specifiers must be provided")]
#[help("{$valid}")]
pub(crate) struct MissingFragmentSpecifier {
    #[primary_span]
    pub span: Span,
    #[suggestion(
        "try adding a specifier here",
        style = "verbose",
        code = ":spec",
        applicability = "maybe-incorrect"
    )]
    pub add_span: Span,
    pub valid: &'static str,
}

#[derive(Diagnostic)]
#[diag("invalid fragment specifier `{$fragment}`")]
#[help("{$help}")]
pub(crate) struct InvalidFragmentSpecifier {
    #[primary_span]
    pub span: Span,
    pub fragment: Ident,
    pub help: &'static str,
}

#[derive(Diagnostic)]
#[diag("expected `(` or `{\"{\"}`, found `{$token}`")]
pub(crate) struct ExpectedParenOrBrace<'a> {
    #[primary_span]
    pub span: Span,
    pub token: Cow<'a, str>,
}

#[derive(Diagnostic)]
#[diag("empty {$kind} delegation is not supported")]
pub(crate) struct EmptyDelegationMac {
    #[primary_span]
    pub span: Span,
    pub kind: String,
}

#[derive(Diagnostic)]
#[diag("glob delegation is only supported in impls")]
pub(crate) struct GlobDelegationOutsideImpls {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("`crate_name` within an `#![cfg_attr]` attribute is forbidden")]
pub(crate) struct CrateNameInCfgAttr {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("`crate_type` within an `#![cfg_attr]` attribute is forbidden")]
pub(crate) struct CrateTypeInCfgAttr {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("qualified path without a trait in glob delegation")]
pub(crate) struct GlobDelegationTraitlessQpath {
    #[primary_span]
    pub span: Span,
}

pub(crate) use metavar_exprs::*;
mod metavar_exprs {
    use super::*;

    #[derive(Diagnostic, Default)]
    #[diag("unexpected trailing tokens")]
    pub(crate) struct MveExtraTokens {
        #[primary_span]
        #[suggestion(
            "try removing {$extra_count ->
                [one] this token
                *[other] these tokens
            }",
            code = "",
            applicability = "machine-applicable"
        )]
        pub span: Span,
        #[label("for this metavariable expression")]
        pub ident_span: Span,
        pub extra_count: usize,

        // The rest is only used for specific diagnostics and can be default if neither
        // `note` is `Some`.
        #[note(
            "the `{$name}` metavariable expression takes {$min_or_exact_args ->
                [zero] no arguments
                [one] a single argument
                *[other] {$min_or_exact_args} arguments
            }"
        )]
        pub exact_args_note: Option<()>,
        #[note(
            "the `{$name}` metavariable expression takes between {$min_or_exact_args} and {$max_args} arguments"
        )]
        pub range_args_note: Option<()>,
        pub min_or_exact_args: usize,
        pub max_args: usize,
        pub name: String,
    }

    #[derive(Diagnostic)]
    #[note("metavariable expressions use function-like parentheses syntax")]
    #[diag("expected `(`")]
    pub(crate) struct MveMissingParen {
        #[primary_span]
        #[label("for this this metavariable expression")]
        pub ident_span: Span,
        #[label("unexpected token")]
        pub unexpected_span: Option<Span>,
        #[suggestion(
            "try adding parentheses",
            code = "( /* ... */ )",
            applicability = "has-placeholders"
        )]
        pub insert_span: Option<Span>,
    }

    #[derive(Diagnostic)]
    #[note("valid metavariable expressions are {$valid_expr_list}")]
    #[diag("unrecognized metavariable expression")]
    pub(crate) struct MveUnrecognizedExpr {
        #[primary_span]
        #[label("not a valid metavariable expression")]
        pub span: Span,
        pub valid_expr_list: &'static str,
    }

    #[derive(Diagnostic)]
    #[diag("variable `{$key}` is not recognized in meta-variable expression")]
    pub(crate) struct MveUnrecognizedVar {
        #[primary_span]
        pub span: Span,
        pub key: MacroRulesNormalizedIdent,
    }
}

#[derive(Diagnostic)]
#[diag("`{$rule_kw}` rule argument matchers require parentheses")]
pub(crate) struct MacroArgsBadDelim {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub sugg: MacroArgsBadDelimSugg,
    pub rule_kw: Symbol,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    "the delimiters should be `(` and `)`",
    applicability = "machine-applicable"
)]
pub(crate) struct MacroArgsBadDelimSugg {
    #[suggestion_part(code = "(")]
    pub open: Span,
    #[suggestion_part(code = ")")]
    pub close: Span,
}

#[derive(LintDiagnostic)]
#[diag("unused doc comment")]
#[help(
    "to document an item produced by a macro, the macro must produce the documentation as part of its expansion"
)]
pub(crate) struct MacroCallUnusedDocComment {
    #[label("rustdoc does not generate documentation for macro invocations")]
    pub span: Span,
}

#[derive(LintDiagnostic)]
#[diag(
    "the meaning of the `pat` fragment specifier is changing in Rust 2021, which may affect this macro"
)]
pub(crate) struct OrPatternsBackCompat {
    #[suggestion(
        "use pat_param to preserve semantics",
        code = "{suggestion}",
        applicability = "machine-applicable"
    )]
    pub span: Span,
    pub suggestion: String,
}

#[derive(LintDiagnostic)]
#[diag("trailing semicolon in macro used in expression position")]
pub(crate) struct TrailingMacro {
    #[note("macro invocations at the end of a block are treated as expressions")]
    #[note(
        "to ignore the value produced by the macro, add a semicolon after the invocation of `{$name}`"
    )]
    pub is_trailing: bool,
    pub name: Ident,
}

#[derive(LintDiagnostic)]
#[diag("unused attribute `{$attr_name}`")]
pub(crate) struct UnusedBuiltinAttribute {
    #[note(
        "the built-in attribute `{$attr_name}` will be ignored, since it's applied to the macro invocation `{$macro_name}`"
    )]
    pub invoc_span: Span,
    pub attr_name: Symbol,
    pub macro_name: String,
    #[suggestion(
        "remove the attribute",
        code = "",
        applicability = "machine-applicable",
        style = "tool-only"
    )]
    pub attr_span: Span,
}
