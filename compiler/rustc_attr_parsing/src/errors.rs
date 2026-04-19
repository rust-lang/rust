use rustc_errors::{DiagArgValue, MultiSpan};
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
