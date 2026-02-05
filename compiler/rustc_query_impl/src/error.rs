use rustc_hir::limit::Limit;
use rustc_macros::{Diagnostic, Subdiagnostic};
use rustc_span::{Span, Symbol};

#[derive(Diagnostic)]
#[help(
    "consider increasing the recursion limit by adding a `#![recursion_limit = \"{$suggested_limit}\"]` attribute to your crate (`{$crate_name}`)"
)]
#[diag("queries overflow the depth limit!")]
pub(crate) struct QueryOverflow {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub note: QueryOverflowNote,
    pub suggested_limit: Limit,
    pub crate_name: Symbol,
}

#[derive(Subdiagnostic)]
#[note("query depth increased by {$depth} when {$desc}")]
pub(crate) struct QueryOverflowNote {
    pub desc: String,
    pub depth: usize,
}
