use rustc_errors::MultiSpan;
use rustc_macros::{Diagnostic, Subdiagnostic};
use rustc_span::{Span, Symbol};

#[derive(Diagnostic)]
#[diag("`{$name}` attribute cannot be used at crate level")]
pub(crate) struct InvalidAttrAtCrateLevel {
    #[primary_span]
    pub span: Span,
    #[suggestion(
        "perhaps you meant to use an outer attribute",
        code = "",
        applicability = "machine-applicable",
        style = "verbose"
    )]
    pub sugg_span: Option<Span>,
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
#[diag("most attributes are not supported in `where` clauses")]
#[help("only `#[cfg]` and `#[cfg_attr]` are supported")]
pub(crate) struct UnsupportedAttributesInWhere {
    #[primary_span]
    pub span: MultiSpan,
}
