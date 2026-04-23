use rustc_errors::{Applicability, DiagArgValue, MultiSpan};
use rustc_macros::{Diagnostic, Subdiagnostic};
use rustc_span::{Span, Symbol};

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
#[diag("unsafe attribute used without unsafe")]
pub(crate) struct UnsafeAttrOutsideUnsafeLint {
    #[label("usage of unsafe attribute")]
    pub span: Span,
    #[subdiagnostic]
    pub suggestion: Option<crate::session_diagnostics::UnsafeAttrOutsideUnsafeSuggestion>,
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
                suggestions.into_iter().map(|s| format!("`{s}`").into()).collect(),
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
pub(crate) struct EmptyAttributeList<'a> {
    #[suggestion(
        "{$valid_without_list ->
            [true] remove these parentheses
            *[other] remove this attribute
        }",
        code = "",
        applicability = "machine-applicable"
    )]
    pub attr_span: Span,
    pub attr_path: &'a str,
    pub valid_without_list: bool,
}

#[derive(Diagnostic)]
#[diag("`#[{$name}]` attribute cannot be used on {$target}")]
#[warning(
    "this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!"
)]
#[help("`#[{$name}]` can {$only}be applied to {$applied}")]
pub(crate) struct InvalidTargetLint {
    pub name: String,
    pub target: &'static str,
    pub applied: DiagArgValue,
    pub only: &'static str,
    #[suggestion(
        "remove the attribute",
        code = "",
        applicability = "machine-applicable",
        style = "tool-only"
    )]
    pub attr_span: Span,
}

#[derive(Diagnostic)]
#[diag(
    "{$is_used_as_inner ->
        [false] crate-level attribute should be an inner attribute: add an exclamation mark: `#![{$name}]`
        *[other] the `#![{$name}]` attribute can only be used at the crate root
    }"
)]
pub(crate) struct InvalidAttrStyle<'a> {
    pub name: &'a str,
    pub is_used_as_inner: bool,
    #[note("this attribute does not have an `!`, which means it is applied to this {$target}")]
    pub target_span: Option<Span>,
    pub target: &'static str,
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
#[diag("`#![doc(auto_cfg({$attr_name}(...)))]` only accepts identifiers or key/value items")]
pub(crate) struct DocAutoCfgHideShowUnexpectedItem {
    pub attr_name: Symbol,
}

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
#[diag("`#[diagnostic::on_const]` can only be applied to non-const trait implementations")]
pub(crate) struct DiagnosticOnConstOnlyForTraitImpls {
    #[label("not a trait implementation")]
    pub target_span: Span,
}

#[derive(Diagnostic)]
#[diag("`#[diagnostic::on_move]` can only be applied to enums, structs or unions")]
pub(crate) struct DiagnosticOnMoveOnlyForAdt;

#[derive(Diagnostic)]
#[diag("`#[diagnostic::on_unimplemented]` can only be applied to trait definitions")]
pub(crate) struct DiagnosticOnUnimplementedOnlyForTraits;

#[derive(Diagnostic)]
#[diag("`#[diagnostic::on_unknown]` can only be applied to `use` statements")]
pub(crate) struct DiagnosticOnUnknownOnlyForImports {
    #[label("not an import")]
    pub target_span: Span,
}

#[derive(Diagnostic)]
#[diag("`#[diagnostic::on_unmatch_args]` can only be applied to macro definitions")]
pub(crate) struct DiagnosticOnUnmatchArgsOnlyForMacros;

#[derive(Diagnostic)]
#[diag("`#[diagnostic::do_not_recommend]` can only be placed on trait implementations")]
pub(crate) struct IncorrectDoNotRecommendLocation {
    #[label("not a trait implementation")]
    pub target_span: Span,
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
#[diag("positional format arguments are not allowed here")]
#[help(
    "only named format arguments with the name of one of the generic types are allowed in this context"
)]
pub(crate) struct DisallowedPositionalArgument;

#[derive(Diagnostic)]
#[diag("format arguments are not allowed here")]
#[help("consider removing this format argument")]
pub(crate) struct DisallowedPlaceholder;

#[derive(Diagnostic)]
#[diag("invalid format specifier")]
#[help("no format specifier are supported in this position")]
pub(crate) struct InvalidFormatSpecifier;

#[derive(Diagnostic)]
#[diag("{$description}")]
pub(crate) struct WrappedParserError<'a> {
    pub description: &'a str,
    #[label("{$label}")]
    pub span: Span,
    pub label: &'a str,
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
