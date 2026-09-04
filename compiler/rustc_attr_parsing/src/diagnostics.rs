use std::num::IntErrorKind;

use rustc_attr_ir::{AttrPath, MirDialect, MirPhase};
use rustc_errors::codes::*;
use rustc_errors::{
    Applicability, Diag, DiagArgValue, DiagCtxtHandle, Diagnostic, E0264, EmissionGuarantee, Level,
    MultiSpan,
};
use rustc_macros::{Diagnostic, Subdiagnostic};
use rustc_span::{Ident, Span, Symbol};
use rustc_target::spec::TargetTuple;

use crate::AttributeTemplate;
use crate::context::Suggestion;

#[derive(Diagnostic)]
#[diag("`{$name}` attribute cannot be used at crate level")]
pub(crate) struct InvalidAttrAtCrateLevel {
    #[primary_span]
    pub span: Span,
    #[suggestion(
        "perhaps you meant to use an outer attribute",
        code = "#[",
        applicability = "machine-applicable",
        style = "verbose"
    )]
    pub pound_to_opening_bracket: Span,
    pub name: Symbol,
    #[subdiagnostic]
    pub item: Option<ItemFollowingInnerAttr>,
}

#[derive(Clone, Copy, Subdiagnostic)]
#[label("the inner attribute doesn't annotate this item")]
pub(crate) struct ItemFollowingInnerAttr {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("invalid edition in edition redirect")]
pub(crate) struct InvalidEditionRedirect {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("unreachable configuration predicate")]
pub(crate) struct UnreachableCfgSelectPredicate {
    #[label("this configuration predicate is never reached")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("most attributes are not supported in `where` clauses")]
#[help("only `#[cfg]` and `#[cfg_attr]` are supported")]
pub(crate) struct UnsupportedAttributesInWhere {
    #[primary_span]
    pub span: MultiSpan,
}

#[derive(Diagnostic)]
#[diag("unreachable configuration predicate")]
pub(crate) struct UnreachableCfgSelectPredicateWildcard {
    #[label("this configuration predicate is never reached")]
    pub span: Span,

    #[label("always matches")]
    pub wildcard_span: Span,
}

#[derive(Diagnostic)]
#[diag("must be a name of an associated function")]
pub(crate) struct MustBeNameOfAssociatedFunction {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("functions names are duplicated")]
#[note("all `#[rustc_must_implement_one_of]` arguments must be unique")]
pub(crate) struct FunctionNamesDuplicated {
    #[primary_span]
    pub spans: Vec<Span>,
}

#[derive(Diagnostic)]
#[diag("unsafe attribute used without unsafe")]
pub(crate) struct UnsafeAttrOutsideUnsafeLint {
    #[label("usage of unsafe attribute")]
    pub span: Span,
    #[subdiagnostic]
    pub suggestion: Option<crate::diagnostics::UnsafeAttrOutsideUnsafeSuggestion>,
}

#[derive(Diagnostic)]
#[diag(
    "{$num_suggestions ->
        [1] attribute must be of the form {$suggestions}
        *[other] valid forms for the attribute are {$suggestions}
    }"
)]
pub(crate) struct IllFormedAttributeInput {
    pub num_suggestions: usize,
    pub suggestions: DiagArgValue,
    #[note("for more information, visit <{$docs}>")]
    pub has_docs: bool,
    pub docs: &'static str,
    #[subdiagnostic]
    help: Option<IllFormedAttributeInputHelp>,
}

impl IllFormedAttributeInput {
    pub(crate) fn new(
        suggestions: &[String],
        docs: Option<&'static str>,
        help: Option<&str>,
    ) -> Self {
        Self {
            num_suggestions: suggestions.len(),
            suggestions: DiagArgValue::StrListSepByAnd(
                suggestions.iter().map(|s| format!("`{s}`").into()).collect(),
            ),
            has_docs: docs.is_some(),
            docs: docs.unwrap_or(""),
            help: help.map(|h| IllFormedAttributeInputHelp { lint: h.to_string() }),
        }
    }
}

#[derive(Subdiagnostic)]
#[help(
    "if you meant to silence a warning, consider using #![allow({$lint})] or #![expect({$lint})]"
)]
struct IllFormedAttributeInputHelp {
    pub lint: String,
}

#[derive(Diagnostic)]
#[diag("unused attribute")]
#[note(
    "{$valid_without_list ->
        [true] using `{$attr_path}` with an empty list is equivalent to not using a list at all
        *[other] using `{$attr_path}` with an empty list has no effect
    }"
)]
pub(crate) struct EmptyAttributeList {
    #[suggestion(
        "{$valid_without_list ->
            [true] remove these parentheses
            *[other] remove this attribute
        }",
        code = "",
        applicability = "machine-applicable"
    )]
    pub attr_span: Span,
    pub attr_path: String,
    pub valid_without_list: bool,
}

#[derive(Diagnostic)]
#[diag(
    "{$is_used_as_inner ->
        [false] crate-level attribute should be an inner attribute: add an exclamation mark: `#![{$name}]`
        *[other] the `#![{$name}]` attribute can only be used at the crate root
    }"
)]
pub(crate) struct InvalidAttrStyle {
    pub name: String,
    pub is_used_as_inner: bool,
    #[note("this attribute does not have an `!`, which means it is applied to this {$target}")]
    pub target_span: Option<Span>,
    pub target: &'static str,
    pub crate_root_path: String,
    #[help("the crate root is at `{$crate_root_path}`")]
    pub show_crate_root_help: bool,
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("doc alias is duplicated")]
pub(crate) struct DocAliasDuplicated {
    #[label("first defined here")]
    pub first_definition: Span,
}

#[derive(Diagnostic)]
#[diag("only `hide` or `show` are allowed in `#[doc(auto_cfg(...))]`")]
pub(crate) struct DocAutoCfgExpectsHideOrShow;

#[derive(Diagnostic)]
#[diag("there exists a built-in attribute with the same name")]
pub(crate) struct AmbiguousDeriveHelpers;

#[derive(Diagnostic)]
#[diag("`#![doc(auto_cfg({$attr_name}(...)))]` only accepts identifiers or `values(...)`")]
pub(crate) struct DocAutoCfgHideShowUnexpectedItem {
    pub attr_name: Symbol,
}

#[derive(Diagnostic)]
#[diag("`any()` was used when other values were provided")]
pub(crate) struct DocAutoCfgHideShowValuesMix {
    #[label("value declared here")]
    pub value_span: Span,
}

#[derive(Diagnostic)]
#[diag("unexpected item after `values()`")]
pub(crate) struct DocAutoCfgHideShowUnexpectedItemAfterValues;

#[derive(Diagnostic)]
#[diag("`#![doc(auto_cfg({$attr_name}(...)))]` expects a list of items")]
pub(crate) struct DocAutoCfgHideShowExpectsList {
    pub attr_name: Symbol,
}

#[derive(Diagnostic)]
#[diag("unknown `doc` attribute `include`")]
pub(crate) struct DocUnknownInclude {
    pub inner: &'static str,
    pub value: Symbol,
    #[suggestion(
        "use `doc = include_str!` instead",
        code = "#{inner}[doc = include_str!(\"{value}\")]"
    )]
    pub sugg: (Span, Applicability),
}

#[derive(Diagnostic)]
#[diag("unknown `doc` attribute `spotlight`")]
#[note("`doc(spotlight)` was renamed to `doc(notable_trait)`")]
#[note("`doc(spotlight)` is now a no-op")]
pub(crate) struct DocUnknownSpotlight {
    #[suggestion(
        "use `notable_trait` instead",
        style = "short",
        applicability = "machine-applicable",
        code = "notable_trait"
    )]
    pub sugg_span: Span,
}

#[derive(Diagnostic)]
#[diag("unknown `doc` attribute `{$name}`")]
#[note(
    "`doc` attribute `{$name}` no longer functions; see issue #44136 <https://github.com/rust-lang/rust/issues/44136>"
)]
#[note("`doc({$name})` is now a no-op")]
pub(crate) struct DocUnknownPasses {
    pub name: Symbol,
    #[label("no longer functions")]
    pub note_span: Span,
}

#[derive(Diagnostic)]
#[diag("unknown `doc` attribute `plugins`")]
#[note(
    "`doc` attribute `plugins` no longer functions; see issue #44136 <https://github.com/rust-lang/rust/issues/44136> and CVE-2018-1000622 <https://nvd.nist.gov/vuln/detail/CVE-2018-1000622>"
)]
#[note("`doc(plugins)` is now a no-op")]
pub(crate) struct DocUnknownPlugins {
    #[label("no longer functions")]
    pub label_span: Span,
}

#[derive(Diagnostic)]
#[diag("unknown `doc` attribute `{$name}`")]
pub(crate) struct DocUnknownAny {
    pub name: Symbol,
}

#[derive(Diagnostic)]
#[diag("expected boolean for `#[doc(auto_cfg = ...)]`")]
pub(crate) struct DocAutoCfgWrongLiteral;

#[derive(Diagnostic)]
#[diag("there must be at least one identifier before `values(...)`")]
pub(crate) struct DocAutoCfgHideShowNoIdentBeforeValues;

#[derive(Diagnostic)]
#[diag("`#[doc(test(...)]` takes a list of attributes")]
pub(crate) struct DocTestTakesList;

#[derive(Diagnostic)]
#[diag("unknown `doc(test)` attribute `{$name}`")]
pub(crate) struct DocTestUnknown {
    pub name: Symbol,
}

#[derive(Diagnostic)]
#[diag("`#![doc(test(...)]` does not take a literal")]
pub(crate) struct DocTestLiteral;

#[derive(Diagnostic)]
#[diag("this attribute can only be applied at the crate level")]
#[note(
    "read <https://doc.rust-lang.org/nightly/rustdoc/the-doc-attribute.html#at-the-crate-level> for more information"
)]
pub(crate) struct AttrCrateLevelOnly;

#[derive(Diagnostic)]
#[diag("`#[diagnostic::do_not_recommend]` does not expect any arguments")]
pub(crate) struct DoNotRecommendDoesNotExpectArgs;

#[derive(Diagnostic)]
#[diag("`#[diagnostic::opaque]` does not expect any arguments")]
pub(crate) struct OpaqueDoesNotExpectArgs;

#[derive(Diagnostic)]
#[diag("invalid `crate_type` value")]
pub(crate) struct UnknownCrateTypes {
    #[subdiagnostic]
    pub sugg: Option<UnknownCrateTypesSuggestion>,
}

#[derive(Subdiagnostic)]
#[suggestion("did you mean", code = r#""{snippet}""#, applicability = "maybe-incorrect")]
pub(crate) struct UnknownCrateTypesSuggestion {
    #[primary_span]
    pub span: Span,
    pub snippet: Symbol,
}

#[derive(Diagnostic)]
#[diag("malformed `doc` attribute input")]
#[warning(
    "this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!"
)]
pub(crate) struct MalformedDoc;

#[derive(Diagnostic)]
#[diag("didn't expect any arguments here")]
#[warning(
    "this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!"
)]
pub(crate) struct ExpectedNoArgs;

#[derive(Diagnostic)]
#[diag("expected this to be of the form `... = \"...\"`")]
#[warning(
    "this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!"
)]
pub(crate) struct ExpectedNameValue;

#[derive(Diagnostic)]
#[diag("malformed `{$attribute}` attribute")]
#[help("{$options}")]
pub(crate) struct MalFormedDiagnosticAttributeLint {
    pub attribute: &'static str,
    pub options: &'static str,
    #[label("invalid option found here")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("{$description}")]
pub(crate) struct WrappedParserError {
    pub description: String,
    #[label("{$label}")]
    pub span: Span,
    pub label: String,
}

#[derive(Diagnostic)]
#[diag("`{$option_name}` is ignored due to previous definition of `{$option_name}`")]
pub(crate) struct IgnoredDiagnosticOption {
    pub option_name: Symbol,
    #[label("`{$option_name}` is first declared here")]
    pub first_span: Span,
    #[label("`{$option_name}` is later redundantly declared here")]
    pub later_span: Span,
}

#[derive(Diagnostic)]
#[diag("missing options for `{$attribute}` attribute")]
#[help("{$options}")]
pub(crate) struct MissingOptionsForDiagnosticAttribute {
    pub attribute: &'static str,
    pub options: &'static str,
}

#[derive(Diagnostic)]
#[diag("expected a literal or missing delimiter")]
#[help(
    "only literals are allowed as values for the `message`, `note` and `label` options. These options must be separated by a comma"
)]
pub(crate) struct NonMetaItemDiagnosticAttribute;

#[derive(Diagnostic, Clone, Copy)]
pub(crate) enum FormatWarning {
    #[diag("positional arguments are not permitted in diagnostic attributes")]
    #[help("you can print empty braces by escaping them")]
    PositionalArgument {
        #[label("remove this format argument")]
        span: Span,
    },

    #[diag("indexed format arguments are not permitted in diagnostic attributes")]
    IndexedArgument {
        #[label("remove this format argument")]
        span: Span,
    },

    #[diag("format specifiers are not permitted in diagnostic attributes")]
    InvalidSpecifier {
        #[label("remove this format specifier")]
        span: Span,
    },

    #[diag("this format argument is not allowed in `#[{$attr}]`")]
    #[note("{$allowed}")]
    DisallowedPlaceholder {
        #[label("remove this format argument")]
        span: Span,
        attr: &'static str,
        allowed: &'static str,
    },
}

#[derive(Subdiagnostic)]
pub(crate) enum UnexpectedCfgCargoHelp {
    #[help("consider using a Cargo feature instead")]
    #[help(
        "or consider adding in `Cargo.toml` the `check-cfg` lint config for the lint:{$cargo_toml_lint_cfg}"
    )]
    LintCfg { cargo_toml_lint_cfg: String },
    #[help("consider using a Cargo feature instead")]
    #[help(
        "or consider adding in `Cargo.toml` the `check-cfg` lint config for the lint:{$cargo_toml_lint_cfg}"
    )]
    #[help("or consider adding `{$build_rs_println}` to the top of the `build.rs`")]
    LintCfgAndBuildRs { cargo_toml_lint_cfg: String, build_rs_println: String },
}

impl UnexpectedCfgCargoHelp {
    fn cargo_toml_lint_cfg(unescaped: &str) -> String {
        format!(
            "\n [lints.rust]\n unexpected_cfgs = {{ level = \"warn\", check-cfg = ['{unescaped}'] }}"
        )
    }

    pub(crate) fn lint_cfg(unescaped: &str) -> Self {
        UnexpectedCfgCargoHelp::LintCfg {
            cargo_toml_lint_cfg: Self::cargo_toml_lint_cfg(unescaped),
        }
    }

    pub(crate) fn lint_cfg_and_build_rs(unescaped: &str, escaped: &str) -> Self {
        UnexpectedCfgCargoHelp::LintCfgAndBuildRs {
            cargo_toml_lint_cfg: Self::cargo_toml_lint_cfg(unescaped),
            build_rs_println: format!("println!(\"cargo::rustc-check-cfg={escaped}\");"),
        }
    }
}

#[derive(Subdiagnostic)]
#[help("to expect this configuration use `{$cmdline_arg}`")]
pub(crate) struct UnexpectedCfgRustcHelp {
    pub cmdline_arg: String,
}

impl UnexpectedCfgRustcHelp {
    pub(crate) fn new(unescaped: &str) -> Self {
        Self { cmdline_arg: format!("--check-cfg={unescaped}") }
    }
}

#[derive(Subdiagnostic)]
#[note(
    "using a cfg inside a {$macro_kind} will use the cfgs from the destination crate and not the ones from the defining crate"
)]
#[help("try referring to `{$macro_name}` crate for guidance on how handle this unexpected cfg")]
pub(crate) struct UnexpectedCfgRustcMacroHelp {
    pub macro_kind: &'static str,
    pub macro_name: Symbol,
}

#[derive(Subdiagnostic)]
#[note(
    "using a cfg inside a {$macro_kind} will use the cfgs from the destination crate and not the ones from the defining crate"
)]
#[help("try referring to `{$macro_name}` crate for guidance on how handle this unexpected cfg")]
pub(crate) struct UnexpectedCfgCargoMacroHelp {
    pub macro_kind: &'static str,
    pub macro_name: Symbol,
}

#[derive(Diagnostic)]
#[diag("unexpected `cfg` condition name: `{$name}`")]
pub(crate) struct UnexpectedCfgName {
    #[subdiagnostic]
    pub code_sugg: unexpected_cfg_name::CodeSuggestion,
    #[subdiagnostic]
    pub invocation_help: unexpected_cfg_name::InvocationHelp,

    pub name: Symbol,
}

pub(crate) mod unexpected_cfg_name {
    use rustc_errors::DiagSymbolList;
    use rustc_macros::Subdiagnostic;
    use rustc_span::{Ident, Span, Symbol};

    #[derive(Subdiagnostic)]
    pub(crate) enum CodeSuggestion {
        #[help("consider defining some features in `Cargo.toml`")]
        DefineFeatures,
        #[multipart_suggestion(
            "there is a similar config predicate: `version(\"..\")`",
            applicability = "machine-applicable"
        )]
        VersionSyntax {
            #[suggestion_part(code = "(")]
            between_name_and_value: Span,
            #[suggestion_part(code = ")")]
            after_value: Span,
        },
        #[suggestion(
            "there is a config with a similar name and value",
            applicability = "maybe-incorrect",
            code = "{code}"
        )]
        SimilarNameAndValue {
            #[primary_span]
            span: Span,
            code: String,
        },
        #[suggestion(
            "there is a config with a similar name and no value",
            applicability = "maybe-incorrect",
            code = "{code}"
        )]
        SimilarNameNoValue {
            #[primary_span]
            span: Span,
            code: String,
        },
        #[suggestion(
            "there is a config with a similar name and different values",
            applicability = "maybe-incorrect",
            code = "{code}"
        )]
        SimilarNameDifferentValues {
            #[primary_span]
            span: Span,
            code: String,
            #[subdiagnostic]
            expected: Option<ExpectedValues>,
        },
        #[suggestion(
            "there is a config with a similar name",
            applicability = "maybe-incorrect",
            code = "{code}"
        )]
        SimilarName {
            #[primary_span]
            span: Span,
            code: String,
            #[subdiagnostic]
            expected: Option<ExpectedValues>,
        },
        SimilarValues {
            #[subdiagnostic]
            with_similar_values: Vec<FoundWithSimilarValue>,
            #[subdiagnostic]
            expected_names: Option<ExpectedNames>,
        },
        #[suggestion(
            "you may have meant to use `{$literal}` (notice the capitalization). Doing so makes this predicate evaluate to `{$literal}` unconditionally",
            applicability = "machine-applicable",
            style = "verbose",
            code = "{literal}"
        )]
        BooleanLiteral {
            #[primary_span]
            span: Span,
            literal: bool,
        },
    }

    #[derive(Subdiagnostic)]
    #[help("expected values for `{$best_match}` are: {$possibilities}")]
    pub(crate) struct ExpectedValues {
        pub best_match: Symbol,
        pub possibilities: DiagSymbolList,
    }

    #[derive(Subdiagnostic)]
    #[suggestion(
        "found config with similar value",
        applicability = "maybe-incorrect",
        code = "{code}"
    )]
    pub(crate) struct FoundWithSimilarValue {
        #[primary_span]
        pub span: Span,
        pub code: String,
    }

    #[derive(Subdiagnostic)]
    #[help_once(
        "expected names are: {$possibilities}{$and_more ->
            [0] {\"\"}
            *[other] {\" \"}and {$and_more} more
        }"
    )]
    pub(crate) struct ExpectedNames {
        pub possibilities: DiagSymbolList<Ident>,
        pub and_more: usize,
    }

    #[derive(Subdiagnostic)]
    pub(crate) enum InvocationHelp {
        #[note(
            "see <https://doc.rust-lang.org/nightly/rustc/check-cfg/cargo-specifics.html> for more information about checking conditional configuration"
        )]
        Cargo {
            #[subdiagnostic]
            macro_help: Option<super::UnexpectedCfgCargoMacroHelp>,
            #[subdiagnostic]
            help: Option<super::UnexpectedCfgCargoHelp>,
        },
        #[note(
            "see <https://doc.rust-lang.org/nightly/rustc/check-cfg.html> for more information about checking conditional configuration"
        )]
        Rustc {
            #[subdiagnostic]
            macro_help: Option<super::UnexpectedCfgRustcMacroHelp>,
            #[subdiagnostic]
            help: super::UnexpectedCfgRustcHelp,
        },
    }
}

#[derive(Diagnostic)]
#[diag(
    "unexpected `cfg` condition value: {$has_value ->
        [true] `{$value}`
        *[false] (none)
    }"
)]
pub(crate) struct UnexpectedCfgValue {
    #[subdiagnostic]
    pub code_sugg: unexpected_cfg_value::CodeSuggestion,
    #[subdiagnostic]
    pub invocation_help: unexpected_cfg_value::InvocationHelp,

    pub has_value: bool,
    pub value: String,
}

pub(crate) mod unexpected_cfg_value {
    use rustc_errors::DiagSymbolList;
    use rustc_macros::Subdiagnostic;
    use rustc_span::{Span, Symbol};

    #[derive(Subdiagnostic)]
    pub(crate) enum CodeSuggestion {
        ChangeValue {
            #[subdiagnostic]
            expected_values: ExpectedValues,
            #[subdiagnostic]
            suggestion: Option<ChangeValueSuggestion>,
        },
        #[note("no expected value for `{$name}`")]
        RemoveValue {
            #[subdiagnostic]
            suggestion: Option<RemoveValueSuggestion>,

            name: Symbol,
        },
        #[note("no expected values for `{$name}`")]
        RemoveCondition {
            #[subdiagnostic]
            suggestion: RemoveConditionSuggestion,

            name: Symbol,
        },
        ChangeName {
            #[subdiagnostic]
            suggestions: Vec<ChangeNameSuggestion>,
        },
    }

    #[derive(Subdiagnostic)]
    pub(crate) enum ChangeValueSuggestion {
        #[suggestion(
            "there is a expected value with a similar name",
            code = r#""{best_match}""#,
            applicability = "maybe-incorrect"
        )]
        SimilarName {
            #[primary_span]
            span: Span,
            best_match: Symbol,
        },
        #[suggestion(
            "specify a config value",
            code = r#" = "{first_possibility}""#,
            applicability = "maybe-incorrect"
        )]
        SpecifyValue {
            #[primary_span]
            span: Span,
            first_possibility: Symbol,
        },
    }

    #[derive(Subdiagnostic)]
    #[suggestion("remove the value", code = "", applicability = "maybe-incorrect")]
    pub(crate) struct RemoveValueSuggestion {
        #[primary_span]
        pub span: Span,
    }

    #[derive(Subdiagnostic)]
    #[suggestion("remove the condition", code = "", applicability = "maybe-incorrect")]
    pub(crate) struct RemoveConditionSuggestion {
        #[primary_span]
        pub span: Span,
    }

    #[derive(Subdiagnostic)]
    #[note(
        "expected values for `{$name}` are: {$have_none_possibility ->
            [true] {\"(none), \"}
            *[false] {\"\"}
        }{$possibilities}{$and_more ->
            [0] {\"\"}
            *[other] {\" \"}and {$and_more} more
        }"
    )]
    pub(crate) struct ExpectedValues {
        pub name: Symbol,
        pub have_none_possibility: bool,
        pub possibilities: DiagSymbolList,
        pub and_more: usize,
    }

    #[derive(Subdiagnostic)]
    #[suggestion(
        "`{$value}` is an expected value for `{$name}`",
        code = "{name}",
        applicability = "maybe-incorrect",
        style = "verbose"
    )]
    pub(crate) struct ChangeNameSuggestion {
        #[primary_span]
        pub span: Span,
        pub name: Symbol,
        pub value: Symbol,
    }

    #[derive(Subdiagnostic)]
    pub(crate) enum InvocationHelp {
        #[note(
            "see <https://doc.rust-lang.org/nightly/rustc/check-cfg/cargo-specifics.html> for more information about checking conditional configuration"
        )]
        Cargo {
            #[subdiagnostic]
            help: Option<CargoHelp>,
            #[subdiagnostic]
            macro_help: Option<super::UnexpectedCfgCargoMacroHelp>,
        },
        #[note(
            "see <https://doc.rust-lang.org/nightly/rustc/check-cfg.html> for more information about checking conditional configuration"
        )]
        Rustc {
            #[subdiagnostic]
            help: Option<super::UnexpectedCfgRustcHelp>,
            #[subdiagnostic]
            macro_help: Option<super::UnexpectedCfgRustcMacroHelp>,
        },
    }

    #[derive(Subdiagnostic)]
    pub(crate) enum CargoHelp {
        #[help("consider adding `{$value}` as a feature in `Cargo.toml`")]
        AddFeature {
            value: Symbol,
        },
        #[help("consider defining some features in `Cargo.toml`")]
        DefineFeatures,
        Other(#[subdiagnostic] super::UnexpectedCfgCargoHelp),
    }
}

#[derive(Diagnostic)]
pub(crate) enum InvalidOnClause {
    #[diag("empty `on`-clause in `#[rustc_on_unimplemented]`")]
    Empty {
        #[primary_span]
        #[label("empty `on`-clause here")]
        span: Span,
    },
    #[diag("expected a single predicate in `not(..)`")]
    ExpectedOnePredInNot {
        #[primary_span]
        #[label("unexpected quantity of predicates here")]
        span: Span,
    },
    #[diag("literals inside `on`-clauses are not supported")]
    UnsupportedLiteral {
        #[primary_span]
        #[label("unexpected literal here")]
        span: Span,
    },
    #[diag("expected an identifier inside this `on`-clause")]
    ExpectedIdentifier {
        #[primary_span]
        #[label("expected an identifier here, not `{$path}`")]
        span: Span,
        path: AttrPath,
    },
    #[diag("this predicate is invalid")]
    InvalidPredicate {
        #[primary_span]
        #[label("expected one of `any`, `all` or `not` here, not `{$invalid_pred}`")]
        span: Span,
        invalid_pred: Symbol,
    },
    #[diag("invalid flag in `on`-clause")]
    InvalidFlag {
        #[primary_span]
        #[label(
            "expected one of the `crate_local`, `direct` or `from_desugaring` flags, not `{$invalid_flag}`"
        )]
        span: Span,
        invalid_flag: Symbol,
    },
}

#[derive(Diagnostic)]
#[diag("usage of the unsafe `{$attr_path}` attribute")]
#[note("{$note}")]
pub(crate) struct UnsafeAttribute {
    pub attr_path: AttrPath,
    pub note: &'static str,
}

#[derive(Diagnostic)]
#[diag("unknown external lang item: `{$lang_item}`", code = E0264)]
pub(crate) struct UnknownExternLangItem {
    #[primary_span]
    pub span: Span,
    pub lang_item: Symbol,
}

#[derive(Diagnostic)]
#[diag("duplicate tool `{$tool}` registered")]
pub(crate) struct DuplicateTool {
    #[primary_span]
    pub(crate) span: Span,
    pub(crate) tool: Ident,
    #[label("already registered here")]
    pub(crate) old_ident_span: Span,
}

#[derive(Diagnostic)]
#[diag("tool `{$tool}` is reserved and cannot be registered")]
pub(crate) struct ToolReserved {
    #[primary_span]
    pub(crate) span: Span,
    pub(crate) tool: Ident,
}

#[derive(Diagnostic)]
#[diag("unknown diagnostic attribute")]
pub(crate) struct UnknownDiagnosticAttribute {
    #[subdiagnostic]
    pub typo: Option<UnknownDiagnosticAttributeTypo>,
}

#[derive(Subdiagnostic)]
#[suggestion(
    "an attribute with a similar name exists",
    style = "verbose",
    code = "{typo_name}",
    applicability = "machine-applicable"
)]
pub(crate) struct UnknownDiagnosticAttributeTypo {
    #[primary_span]
    pub span: Span,
    pub typo_name: Symbol,
}

#[derive(Diagnostic)]
#[diag("unknown diagnostic attribute")]
pub(crate) struct UnstableDiagnosticAttribute {
    #[note("this is an experimental diagnostic attribute")]
    #[help("add `#![feature({$feature})]` to the crate attributes to enable")]
    pub nightly_build: bool,
    pub feature: Symbol,
}

#[derive(Diagnostic)]
#[diag("`#[rustc_force_inline]` and `#[inline]` cannot be used together")]
pub(crate) struct InlineForceInlineConflict {
    #[primary_span]
    pub force_inline_span: Span,
    #[label("the inline attribute is specified here")]
    pub inline_span: Span,
}

#[derive(Diagnostic)]
#[diag("`#[ffi_const]` function cannot be `#[ffi_pure]`", code = E0757)]
pub(crate) struct BothFfiConstAndPure {
    #[primary_span]
    pub attr_span: Span,
}

#[derive(Diagnostic)]
#[diag("attribute should be applied to `#[repr(transparent)]` types")]
pub(crate) struct RustcPubTransparent {
    #[primary_span]
    pub attr_span: Span,
    #[label("not a `#[repr(transparent)]` type")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("attribute should be applied to a macro")]
pub(crate) struct MacroOnlyAttribute {
    #[primary_span]
    pub attr_span: Span,
    #[label("not a macro")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("{$attr_str} attribute cannot have empty value")]
pub(crate) struct DocAliasEmpty<'a> {
    #[primary_span]
    pub span: Span,
    pub attr_str: &'a str,
}

#[derive(Diagnostic)]
#[diag("{$char_} character isn't allowed in {$attr_str}")]
pub(crate) struct DocAliasBadChar<'a> {
    #[primary_span]
    pub span: Span,
    pub attr_str: &'a str,
    pub char_: char,
}

#[derive(Diagnostic)]
#[diag("{$attr_str} cannot start or end with ' '")]
pub(crate) struct DocAliasStartEnd<'a> {
    #[primary_span]
    pub span: Span,
    pub attr_str: &'a str,
}

#[derive(Diagnostic)]
#[diag("`#[{$name})]` is missing a `{$field}` argument")]
pub(crate) struct CguFieldsMissing<'a> {
    #[primary_span]
    pub span: Span,
    pub name: &'a AttrPath,
    pub field: Symbol,
}

#[derive(Diagnostic)]
#[diag("`#![doc({$attr_name} = \"...\")]` isn't allowed as a crate-level attribute")]
pub(crate) struct DocAttrNotCrateLevel {
    #[primary_span]
    pub span: Span,
    pub attr_name: Symbol,
}

#[derive(Diagnostic)]
#[diag("nonexistent keyword `{$keyword}` used in `#[doc(keyword = \"...\")]`")]
#[help("only existing keywords are allowed in core/std")]
pub(crate) struct DocKeywordNotKeyword {
    #[primary_span]
    pub span: Span,
    pub keyword: Symbol,
}

#[derive(Diagnostic)]
#[diag("nonexistent builtin attribute `{$attribute}` used in `#[doc(attribute = \"...\")]`")]
#[help("only existing builtin attributes are allowed in core/std")]
pub(crate) struct DocAttributeNotAttribute {
    #[primary_span]
    pub span: Span,
    pub attribute: Symbol,
}

#[derive(Diagnostic)]
#[diag(
    "`#[target_feature]` cannot be applied to a {$kind ->
        [panic_handler] `#[panic_handler]`
        *[other] lang item
    } function"
)]
pub(crate) struct TargetFeatureOnLangItem {
    #[primary_span]
    pub attr_span: Span,
    pub kind: Symbol,
    #[label(
        "{$kind ->
            [panic_handler] `#[panic_handler]`
            *[other] lang item
        } function is not allowed to have `#[target_feature]`"
    )]
    pub item_span: Span,
}

#[derive(Diagnostic)]
#[diag(
    "{$name ->
    [panic_impl] `#[panic_handler]`
    *[other] `{$name}` lang item
} function is not allowed to have `#[track_caller]`"
)]
pub(crate) struct TrackCallerOnLangItem {
    #[primary_span]
    pub attr_span: Span,
    pub name: Symbol,
    #[label(
        "{$name ->
            [panic_impl] `#[panic_handler]`
            *[other] `{$name}` lang item
        } function is not allowed to have `#[track_caller]`"
    )]
    pub sig_span: Span,
}

#[derive(Diagnostic)]
#[diag("missing 'since'", code = E0542)]
pub(crate) struct MissingSince {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("missing 'note'", code = E0543)]
pub(crate) struct MissingNote {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("multiple stability levels", code = E0544)]
pub(crate) struct MultipleStabilityLevels {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("`issue` must be a non-zero numeric string or \"none\"", code = E0545)]
pub(crate) struct InvalidIssueString {
    #[primary_span]
    pub span: Span,

    #[subdiagnostic]
    pub cause: Option<InvalidIssueStringCause>,
}

// The error kinds of `IntErrorKind` are duplicated here in order to allow the messages to be
// translatable.
#[derive(Subdiagnostic)]
pub(crate) enum InvalidIssueStringCause {
    #[label("`issue` must not be \"0\", use \"none\" instead")]
    MustNotBeZero {
        #[primary_span]
        span: Span,
    },

    #[label("cannot parse integer from empty string")]
    Empty {
        #[primary_span]
        span: Span,
    },

    #[label("invalid digit found in string")]
    InvalidDigit {
        #[primary_span]
        span: Span,
    },

    #[label("number too large to fit in target type")]
    PosOverflow {
        #[primary_span]
        span: Span,
    },

    #[label("number too small to fit in target type")]
    NegOverflow {
        #[primary_span]
        span: Span,
    },
}

impl InvalidIssueStringCause {
    pub(crate) fn from_int_error_kind(span: Span, kind: &IntErrorKind) -> Option<Self> {
        match kind {
            IntErrorKind::Empty => Some(Self::Empty { span }),
            IntErrorKind::InvalidDigit => Some(Self::InvalidDigit { span }),
            IntErrorKind::PosOverflow => Some(Self::PosOverflow { span }),
            IntErrorKind::NegOverflow => Some(Self::NegOverflow { span }),
            IntErrorKind::Zero => Some(Self::MustNotBeZero { span }),
            _ => None,
        }
    }
}

#[derive(Diagnostic)]
#[diag("missing 'feature'", code = E0546)]
pub(crate) struct MissingFeature {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("'feature' is not an identifier", code = E0546)]
pub(crate) struct NonIdentFeature {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("missing 'issue'", code = E0547)]
pub(crate) struct MissingIssue {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("`rustc_promotable` attribute must be paired with either a `rustc_const_unstable` or a `rustc_const_stable` attribute", code = E0717)]
pub(crate) struct RustcPromotablePairing {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("`rustc_allowed_through_unstable_modules` attribute must be paired with a `stable` attribute", code = E0789)]
pub(crate) struct RustcAllowedUnstablePairing {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("suggestions on deprecated items are unstable")]
pub(crate) struct DeprecatedItemSuggestion {
    #[primary_span]
    pub span: Span,

    #[help("add `#![feature(deprecated_suggestion)]` to the crate root")]
    pub is_nightly: bool,

    #[note("see #94785 for more details")]
    pub details: (),
}

#[derive(Diagnostic)]
#[diag("this `#[deprecated]` annotation has no effect")]
pub(crate) struct DeprecatedAnnotationHasNoEffect {
    #[suggestion(
        "remove the unnecessary deprecation attribute",
        applicability = "machine-applicable",
        code = ""
    )]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("expected single version literal")]
pub(crate) struct ExpectedSingleVersionLiteral {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("expected a version literal")]
pub(crate) struct ExpectedVersionLiteral {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("`{$name}` expects a list of feature names")]
pub(crate) struct ExpectsFeatureList {
    #[primary_span]
    pub span: Span,

    pub name: String,
}

#[derive(Diagnostic)]
#[diag("`{$name}` expects feature names")]
pub(crate) struct ExpectsFeatures {
    #[primary_span]
    pub span: Span,

    pub name: String,
}

#[derive(Diagnostic)]
#[diag("'since' must be a Rust version number, such as \"1.31.0\"")]
pub(crate) struct InvalidSince {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("unknown version literal format, assuming it refers to a future version")]
pub(crate) struct UnknownVersionLiteral {
    #[primary_span]
    pub span: Span,
}

// FIXME(jdonszelmann) duplicated from `rustc_passes`, remove once `check_attr` is integrated.
#[derive(Diagnostic)]
#[diag("multiple `{$name}` attributes")]
pub(crate) struct UnusedMultiple {
    #[primary_span]
    #[suggestion("remove this attribute", code = "", applicability = "machine-applicable")]
    pub this: Span,
    #[note("attribute also specified here")]
    pub other: Span,
    pub name: Symbol,
}

#[derive(Diagnostic)]
#[diag("`export_name` may not be empty")]
pub(crate) struct EmptyExportName {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("`section` may not be empty")]
pub(crate) struct EmptySection {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("`export_name` may not contain null characters", code = E0648)]
pub(crate) struct NullOnExport {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("`link_section` may not contain null characters", code = E0648)]
pub(crate) struct NullOnLinkSection {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("link name may not contain null characters", code = E0648)]
pub(crate) struct NullOnLinkName {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("`objc::class!` may not contain null characters")]
pub(crate) struct NullOnObjcClass {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("`objc::selector!` may not contain null characters")]
pub(crate) struct NullOnObjcSelector {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("`section` may not contain null characters", code = E0648)]
pub(crate) struct NullOnSection {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("`objc::class!` expected a string literal")]
pub(crate) struct ObjcClassExpectedStringLiteral {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("`objc::selector!` expected a string literal")]
pub(crate) struct ObjcSelectorExpectedStringLiteral {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("expected at least one confusable name")]
pub(crate) struct EmptyConfusables {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[help("the `{$name}{$attribute_args}` attribute can {$only}be applied to {$applied}")]
#[diag("the `{$name}{$attribute_args}` attribute cannot be used on {$target}")]
pub(crate) struct InvalidTarget {
    #[primary_span]
    pub span: Span,
    #[suggestion(
        "remove the attribute",
        code = "",
        applicability = "machine-applicable",
        style = "tool-only"
    )]
    pub attr_span: Span,
    pub name: AttrPath,
    pub target: &'static str,
    pub applied: DiagArgValue,
    pub only: &'static str,
    pub attribute_args: String,
    #[subdiagnostic]
    pub help: Option<InvalidTargetHelp>,
    #[warning(
        "this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!"
    )]
    pub previously_accepted: bool,
    #[note(
        "placing this attribute on a macro invocation does nothing even if the macro expands to what would be a valid target for the attribute"
    )]
    pub on_macro_call: bool,
}

#[derive(Subdiagnostic)]
pub(crate) enum InvalidTargetHelp {
    #[multipart_suggestion(
        "did you mean to use `#[export_name]`?",
        applicability = "maybe-incorrect"
    )]
    UseExportName {
        #[suggestion_part(code = "unsafe(")]
        unsafe_open: Option<Span>,
        #[suggestion_part(code = "export_name")]
        name: Span,
        #[suggestion_part(code = ")")]
        unsafe_close: Option<Span>,
    },
    #[help("use `#[rustc_align(...)]` instead")]
    UseRustcAlign,
    #[help("use `#[rustc_align_static(...)]` instead")]
    UseRustcAlignStatic,
}

#[derive(Diagnostic)]
#[diag("invalid alignment value: {$error_part}", code = E0589)]
pub(crate) struct InvalidAlignmentValue {
    #[primary_span]
    pub span: Span,
    pub error_part: String,
}

#[derive(Diagnostic)]
#[diag("item annotated with `#[unstable_feature_bound]` should not be stable")]
#[help(
    "if this item is meant to be stable, do not use any functions annotated with `#[unstable_feature_bound]`. Otherwise, mark this item as unstable with `#[unstable]`"
)]
pub(crate) struct UnstableFeatureBoundIncompatibleStability {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("attribute incompatible with `#[unsafe(naked)]`", code = E0736)]
pub(crate) struct NakedFunctionIncompatibleAttribute {
    #[primary_span]
    #[label("the `{$attr}` attribute is incompatible with `#[unsafe(naked)]`")]
    pub span: Span,
    #[label("function marked with `#[unsafe(naked)]` here")]
    pub naked_span: Span,
    pub attr: String,
}

#[derive(Diagnostic)]
#[diag("ordinal value in `link_ordinal` is too large: `{$ordinal}`")]
#[note("the value may not exceed `u16::MAX`")]
pub(crate) struct LinkOrdinalOutOfRange {
    #[primary_span]
    pub span: Span,
    pub ordinal: u128,
}

#[derive(Diagnostic)]
#[diag("element count in `rustc_scalable_vector` is too large: `{$n}`")]
#[note("the value may not exceed `u16::MAX`")]
pub(crate) struct RustcScalableVectorCountOutOfRange {
    #[primary_span]
    pub span: Span,
    pub n: u128,
}

#[derive(Diagnostic)]
#[diag("attribute requires {$opt} to be enabled")]
pub(crate) struct AttributeRequiresOpt {
    #[primary_span]
    pub span: Span,
    pub opt: &'static str,
}

pub(crate) enum AttributeParseErrorReason<'a> {
    ExpectedNoArgs,
    ExpectedStringLiteral {
        byte_string: Option<Span>,
    },
    ExpectedFilenameLiteral,
    ExpectedIntegerLiteral,
    ExpectedIntegerLiteralInRange {
        lower_bound: isize,
        upper_bound: isize,
    },
    ExpectedAtLeastOneArgument,
    ExpectedArgument,
    ExpectedSingleArgument,
    ExpectedList,
    ExpectedListOrNoArgs,
    ExpectedListWithNumArgsOrMore {
        args: usize,
    },
    ExpectedNameValueOrNoArgs,
    ExpectedNonEmptyStringLiteral,
    ExpectedNotLiteral,
    ExpectedNameValue(Option<Symbol>),
    MissingNameValue(Symbol),
    DuplicateKey(Symbol),
    ExpectedSpecificArgument {
        possibilities: &'a [Symbol],
        strings: bool,
        /// Should we tell the user to write a list when they didn't?
        list: bool,
    },
    ExpectedIdentifier,
}

/// A description of a thing that can be parsed using an attribute parser.
#[derive(Copy, Clone)]
pub enum ParsedDescription {
    /// Used when parsing attributes.
    Attribute,
    /// Used when parsing some macros, such as the `cfg!()` macro.
    Macro,
}

pub(crate) struct AttributeParseError<'a> {
    pub(crate) span: Span,
    pub(crate) inner_span: Span,
    pub(crate) template: AttributeTemplate,
    pub(crate) path: AttrPath,
    pub(crate) description: ParsedDescription,
    pub(crate) reason: AttributeParseErrorReason<'a>,
    pub(crate) suggestions: AttributeParseErrorSuggestions,
}

pub(crate) enum AttributeParseErrorSuggestions {
    CreatedByTemplate(Vec<String>),
    CreatedByParser(Vec<Suggestion>),
}

impl<'a> AttributeParseError<'a> {
    fn render_expected_specific_argument<G>(
        &self,
        diag: &mut Diag<'_, G>,
        possibilities: &[Symbol],
        strings: bool,
    ) where
        G: EmissionGuarantee,
    {
        let quote = if strings { '"' } else { '`' };
        match possibilities {
            &[] => {}
            &[x] => {
                diag.span_label(
                    self.span,
                    format!("the only valid argument here is {quote}{x}{quote}"),
                );
            }
            [first, second] => {
                diag.span_label(
                    self.span,
                    format!("valid arguments are {quote}{first}{quote} or {quote}{second}{quote}"),
                );
            }
            [first @ .., second_to_last, last] => {
                let mut res = String::new();
                for i in first {
                    res.push_str(&format!("{quote}{i}{quote}, "));
                }
                res.push_str(&format!("{quote}{second_to_last}{quote} or {quote}{last}{quote}"));

                diag.span_label(self.span, format!("valid arguments are {res}"));
            }
        }
    }

    fn render_expected_specific_argument_list<G>(
        &self,
        diag: &mut Diag<'_, G>,
        possibilities: &[Symbol],
        strings: bool,
    ) where
        G: EmissionGuarantee,
    {
        let description = self.description();

        let quote = if strings { '"' } else { '`' };
        match possibilities {
            &[] => {}
            &[x] => {
                diag.span_label(
                    self.span,
                    format!(
                        "this {description} is only valid with {quote}{x}{quote} as an argument"
                    ),
                );
            }
            [first, second] => {
                diag.span_label(self.span, format!("this {description} is only valid with either {quote}{first}{quote} or {quote}{second}{quote} as an argument"));
            }
            [first @ .., second_to_last, last] => {
                let mut res = String::new();
                for i in first {
                    res.push_str(&format!("{quote}{i}{quote}, "));
                }
                res.push_str(&format!("{quote}{second_to_last}{quote} or {quote}{last}{quote}"));

                diag.span_label(self.span, format!("this {description} is only valid with one of the following arguments: {res}"));
            }
        }
    }

    fn render_suggestions<G>(&self, diag: &mut Diag<'_, G>)
    where
        G: EmissionGuarantee,
    {
        let description = self.description();

        match &self.suggestions {
            AttributeParseErrorSuggestions::CreatedByTemplate(suggestions) => {
                diag.span_suggestions(
                    self.inner_span,
                    if suggestions.len() == 1 {
                        "must be of the form".to_string()
                    } else {
                        format!(
                            "try changing it to one of the following valid forms of the {description}"
                        )
                    },
                    suggestions.iter().cloned(),
                    Applicability::HasPlaceholders,
                );
            }

            AttributeParseErrorSuggestions::CreatedByParser(suggestions) => {
                for Suggestion { msg, sp, code } in suggestions {
                    diag.span_suggestion_verbose(
                        *sp,
                        msg.clone(),
                        code.clone(),
                        Applicability::MaybeIncorrect,
                    );
                }
            }
        }
    }

    fn description(&self) -> &'static str {
        match self.description {
            ParsedDescription::Attribute => "attribute",
            ParsedDescription::Macro => "macro",
        }
    }
}

impl AttributeParseErrorSuggestions {
    fn len(&self) -> usize {
        match self {
            AttributeParseErrorSuggestions::CreatedByTemplate(items) => items.len(),
            AttributeParseErrorSuggestions::CreatedByParser(items) => items.len(),
        }
    }
}

impl<'a, G: EmissionGuarantee> Diagnostic<'a, G> for AttributeParseError<'_> {
    fn into_diag(self, dcx: DiagCtxtHandle<'a>, level: Level) -> Diag<'a, G> {
        let name = self.path.to_string();

        let description = self.description();

        let mut diag = Diag::new(dcx, level, format!("malformed `{name}` {description} input"));
        diag.span(self.inner_span);
        diag.code(E0539);
        match &self.reason {
            AttributeParseErrorReason::ExpectedStringLiteral { byte_string } => {
                if let Some(start_point_span) = byte_string {
                    diag.span_suggestion(
                        *start_point_span,
                        "consider removing the prefix",
                        "",
                        Applicability::MaybeIncorrect,
                    );
                    diag.note("expected a normal string literal, not a byte string literal");

                    // Avoid emitting an "attribute must be of the form" suggestion, as the
                    // attribute is likely to be well-formed already.
                    return diag;
                }
                diag.span_label(self.span, "expected a string literal here");
            }
            AttributeParseErrorReason::ExpectedFilenameLiteral => {
                diag.span_label(self.span, "expected a filename string literal here");
            }
            AttributeParseErrorReason::ExpectedIntegerLiteral => {
                diag.span_label(self.span, "expected an integer literal here");
            }
            AttributeParseErrorReason::ExpectedIntegerLiteralInRange {
                lower_bound,
                upper_bound,
            } => {
                diag.span_label(
                    self.span,
                    format!(
                        "expected an integer literal in the range of {lower_bound}..={upper_bound}"
                    ),
                );
            }
            AttributeParseErrorReason::ExpectedSingleArgument => {
                diag.span_label(self.span, "expected a single argument here");
                diag.code(E0805);
            }
            AttributeParseErrorReason::ExpectedArgument => {
                diag.span_label(self.span, "expected an argument here");
                diag.code(E0805);
            }
            AttributeParseErrorReason::ExpectedAtLeastOneArgument => {
                diag.span_label(self.span, "expected at least 1 argument here");
            }
            AttributeParseErrorReason::ExpectedList => {
                diag.span_label(self.span, "expected this to be a list");
            }
            AttributeParseErrorReason::ExpectedListOrNoArgs => {
                diag.span_label(self.span, "expected a list or no arguments here");
            }
            AttributeParseErrorReason::ExpectedListWithNumArgsOrMore { args } => {
                diag.span_label(self.span, format!("expected {args} or more items"));
            }
            AttributeParseErrorReason::ExpectedNameValueOrNoArgs => {
                diag.span_label(self.span, "didn't expect a list here");
            }
            AttributeParseErrorReason::ExpectedNonEmptyStringLiteral => {
                diag.span_label(self.span, "string is not allowed to be empty");
            }
            AttributeParseErrorReason::DuplicateKey(key) => {
                diag.span_label(self.span, format!("found `{key}` used as a key more than once"));
                diag.code(E0538);
            }
            AttributeParseErrorReason::ExpectedNotLiteral => {
                diag.span_label(self.span, "didn't expect a literal here");
                diag.code(E0565);
            }
            AttributeParseErrorReason::ExpectedNoArgs => {
                diag.span_label(self.span, "didn't expect any arguments here");
                diag.code(E0565);
            }
            AttributeParseErrorReason::ExpectedNameValue(None) => {
                // If the span is the entire attribute inner, the suggestion we add below this
                // match already contains enough information.
                if self.span != self.inner_span {
                    diag.span_label(self.span, "expected this to be of the form `... = \"...\"`");
                }
            }
            AttributeParseErrorReason::ExpectedNameValue(Some(name)) => {
                diag.span_label(
                    self.span,
                    format!("expected this to be of the form `{name} = \"...\"`"),
                );
            }
            AttributeParseErrorReason::MissingNameValue(name) => {
                diag.span_label(self.span, format!("missing argument `{name} = \"...\"`"));
            }
            AttributeParseErrorReason::ExpectedSpecificArgument {
                possibilities,
                strings,
                list: false,
            } => {
                self.render_expected_specific_argument(&mut diag, possibilities, *strings);
            }
            AttributeParseErrorReason::ExpectedSpecificArgument {
                possibilities,
                strings,
                list: true,
            } => {
                self.render_expected_specific_argument_list(&mut diag, possibilities, *strings);
            }
            AttributeParseErrorReason::ExpectedIdentifier => {
                diag.span_label(self.span, "expected a valid identifier here");
                diag.code(E0565);
            }
        }

        if let Some(link) = self.template.docs {
            diag.note(format!("for more information, visit <{link}>"));
        }

        if self.suggestions.len() < 4 {
            self.render_suggestions(&mut diag);
        }

        diag
    }
}

#[derive(Diagnostic)]
#[diag("`{$name}` is not an unsafe attribute")]
#[note("extraneous unsafe is not allowed in attributes")]
pub(crate) struct InvalidAttrUnsafe {
    #[primary_span]
    #[label("this is not an unsafe attribute")]
    pub span: Span,
    pub name: AttrPath,
}

#[derive(Diagnostic)]
#[diag("unsafe attribute used without unsafe")]
pub(crate) struct UnsafeAttrOutsideUnsafe {
    #[primary_span]
    #[label("usage of unsafe attribute")]
    pub span: Span,
    #[subdiagnostic]
    pub suggestion: Option<UnsafeAttrOutsideUnsafeSuggestion>,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion("wrap the attribute in `unsafe(...)`", applicability = "machine-applicable")]
pub(crate) struct UnsafeAttrOutsideUnsafeSuggestion {
    #[suggestion_part(code = "unsafe(")]
    pub left: Span,
    #[suggestion_part(code = ")")]
    pub right: Span,
}

#[derive(Diagnostic)]
#[diag("wrong meta list delimiters")]
pub(crate) struct MetaBadDelim {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub sugg: MetaBadDelimSugg,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    "the delimiters should be `(` and `)`",
    applicability = "machine-applicable"
)]
pub(crate) struct MetaBadDelimSugg {
    #[suggestion_part(code = "(")]
    pub open: Span,
    #[suggestion_part(code = ")")]
    pub close: Span,
}

#[derive(Diagnostic)]
#[diag("expected a literal (`1u8`, `1.0f32`, `\"string\"`, etc.) here, found {$descr}")]
pub(crate) struct InvalidMetaItem {
    #[primary_span]
    pub span: Span,
    pub descr: String,
    #[subdiagnostic]
    pub quote_ident_sugg: Option<InvalidMetaItemQuoteIdentSugg>,
    #[subdiagnostic]
    pub remove_neg_sugg: Option<InvalidMetaItemRemoveNegSugg>,
    #[label("{$descr}s are not allowed here")]
    pub label: Option<Span>,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    "surround the identifier with quotation marks to make it into a string literal",
    applicability = "machine-applicable"
)]
pub(crate) struct InvalidMetaItemQuoteIdentSugg {
    #[suggestion_part(code = "\"")]
    pub before: Span,
    #[suggestion_part(code = "\"")]
    pub after: Span,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    "negative numbers are not literals, try removing the `-` sign",
    applicability = "machine-applicable"
)]
pub(crate) struct InvalidMetaItemRemoveNegSugg {
    #[suggestion_part(code = "")]
    pub negative_sign: Span,
}

#[derive(Diagnostic)]
#[diag("suffixed literals are not allowed in attributes")]
#[help(
    "instead of using a suffixed literal (`1u8`, `1.0f32`, etc.), use an unsuffixed version (`1`, `1.0`, etc.)"
)]
pub(crate) struct SuffixedLiteralInAttribute {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("link name must not be empty", code = E0454)]
pub(crate) struct EmptyLinkName {
    #[primary_span]
    #[label("empty link name")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("link kind `framework` is only supported on Apple targets", code = E0455)]
pub(crate) struct LinkFrameworkApple {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("`wasm_import_module` is incompatible with other arguments in `#[link]` attributes")]
pub(crate) struct IncompatibleWasmLink {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("`#[link]` attribute requires a `name = \"string\"` argument", code = E0459)]
pub(crate) struct LinkRequiresName {
    #[primary_span]
    #[label("missing `name` argument")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("link kind `raw-dylib` is only supported on Windows targets", code = E0455)]
pub(crate) struct RawDylibOnlyWindows {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(
    "invalid linking modifier syntax, expected '+' or '-' prefix before one of: bundle, verbatim, whole-archive, as-needed, export-symbols"
)]
pub(crate) struct InvalidLinkModifier {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("multiple `{$modifier}` modifiers in a single `modifiers` argument")]
pub(crate) struct MultipleModifiers {
    #[primary_span]
    pub span: Span,
    pub modifier: Symbol,
}

#[derive(Diagnostic)]
#[diag("import name type is only supported on x86")]
pub(crate) struct ImportNameTypeX86 {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("linking modifier `bundle` is only compatible with `static` linking kind")]
pub(crate) struct BundleNeedsStatic {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("linking modifier `export-symbols` is only compatible with `static` linking kind")]
pub(crate) struct ExportSymbolsNeedsStatic {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("linking modifier `whole-archive` is only compatible with `static` linking kind")]
pub(crate) struct WholeArchiveNeedsStatic {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(
    "linking modifier `as-needed` is only compatible with `dylib`, `framework` and `raw-dylib` linking kinds"
)]
pub(crate) struct AsNeededCompatibility {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("import name type can only be used with link kind `raw-dylib`")]
pub(crate) struct ImportNameTypeRaw {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("`limit` must be a non-negative integer")]
pub(crate) struct LimitInvalid<'a> {
    #[primary_span]
    pub span: Span,
    #[label("{$error_str}")]
    pub value_span: Span,
    pub error_str: &'a str,
}

#[derive(Diagnostic)]
#[diag("wrong `cfg_attr` delimiters")]
pub(crate) struct CfgAttrBadDelim {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub sugg: MetaBadDelimSugg,
}

#[derive(Diagnostic)]
#[diag(
    "doc alias attribute expects a string `#[doc(alias = \"a\")]` or a list of strings `#[doc(alias(\"a\", \"b\"))]`"
)]
pub(crate) struct DocAliasMalformed {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("definition of an unknown lang item: `{$name}`", code = E0522)]
pub(crate) struct UnknownLangItem {
    #[primary_span]
    #[label("definition of unknown lang item `{$name}`")]
    pub span: Span,
    pub name: Symbol,
}

#[derive(Diagnostic)]
#[diag("target `{$current_target}` does not support `#[instruction_set({$instruction_set}::*)]`")]
pub(crate) struct UnsupportedInstructionSet<'a> {
    #[primary_span]
    pub span: Span,
    pub instruction_set: Symbol,
    pub current_target: &'a TargetTuple,
}

#[derive(Diagnostic)]
#[diag("`dialect` key required")]
pub(crate) struct CustomMirPhaseRequiresDialect {
    #[primary_span]
    pub attr_span: Span,
    #[label("`phase` argument requires a `dialect` argument")]
    pub phase_span: Span,
}

#[derive(Diagnostic)]
#[diag("the {$dialect} dialect is not compatible with the {$phase} phase")]
pub(crate) struct CustomMirIncompatibleDialectAndPhase {
    pub dialect: MirDialect,
    pub phase: MirPhase,
    #[primary_span]
    pub attr_span: Span,
    #[label("this dialect...")]
    pub dialect_span: Span,
    #[label("... is not compatible with this phase")]
    pub phase_span: Span,
}

#[derive(Diagnostic)]
#[diag("can't mark as unstable using an already stable feature")]
pub(crate) struct UnstableAttrForAlreadyStableFeature {
    #[primary_span]
    #[label("this feature is already stable")]
    #[help("consider removing the attribute")]
    pub attr_span: Span,
    #[label("the stability attribute annotates this item")]
    pub item_span: Span,
}

#[derive(Diagnostic)]
#[diag("invalid Mach-O section specifier")]
pub(crate) struct InvalidMachoSection {
    #[primary_span]
    #[label("not a valid Mach-O section specifier")]
    pub name_span: Span,
    #[subdiagnostic]
    pub reason: InvalidMachoSectionReason,
}

#[derive(Subdiagnostic)]
pub(crate) enum InvalidMachoSectionReason {
    #[note("a Mach-O section specifier requires a segment and a section, separated by a comma")]
    #[help("an example of a valid Mach-O section specifier is `__TEXT,__cstring`")]
    MissingSection,
    #[note("section name `{$section}` is longer than 16 bytes")]
    SectionTooLong { section: String },
}

#[derive(Diagnostic)]
#[diag("`#[sanitize({$field} = ...)]` attribute cannot be used on statics")]
#[help("`#[sanitize]` can be used on statics if only the address is sanitized")]
pub(crate) struct SanitizeInvalidStatic {
    #[primary_span]
    pub span: Span,
    pub field: &'static str,
}

#[derive(Diagnostic)]
#[diag("attribute items not separated with `,`")]
pub(crate) struct ExpectedComma {
    #[primary_span]
    #[suggestion(
        "try adding `,` here",
        code = ",",
        applicability = "maybe-incorrect",
        style = "short"
    )]
    pub span: Span,
    #[subdiagnostic]
    pub additional: Vec<AdditionalCommaSuggestion>,
}

#[derive(Subdiagnostic)]
#[suggestion("try adding `,` here", code = ",", applicability = "maybe-incorrect", style = "short")]
pub(crate) struct AdditionalCommaSuggestion {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("unused attribute")]
pub(crate) struct UnusedDuplicate {
    #[suggestion("remove this attribute", code = "", applicability = "machine-applicable")]
    pub this: Span,
    #[note("attribute also specified here")]
    pub other: Span,
    #[warning(
        "this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!"
    )]
    pub warning: bool,
}
