use rustc_errors::codes::*;
use rustc_hir::limit::Limit;
use rustc_macros::{Diagnostic, Subdiagnostic};
use rustc_span::{Span, Symbol};

#[derive(Subdiagnostic)]
#[note("...which requires {$desc}...")]
pub(crate) struct CycleStack {
    #[primary_span]
    pub span: Span,
    pub desc: String,
}

#[derive(Subdiagnostic)]
pub(crate) enum StackCount {
    #[note("...which immediately requires {$stack_bottom} again")]
    Single,
    #[note("...which again requires {$stack_bottom}, completing the cycle")]
    Multiple,
}

#[derive(Subdiagnostic)]
pub(crate) enum Alias {
    #[note("type aliases cannot be recursive")]
    #[help("consider using a struct, enum, or union instead to break the cycle")]
    #[help(
        "see <https://doc.rust-lang.org/reference/types.html#recursive-types> for more information"
    )]
    Ty,
    #[note("trait aliases cannot be recursive")]
    Trait,
}

#[derive(Subdiagnostic)]
#[note("cycle used when {$usage}")]
pub(crate) struct CycleUsage {
    #[primary_span]
    pub span: Span,
    pub usage: String,
}

#[derive(Diagnostic)]
#[diag("cycle detected when {$stack_bottom}", code = E0391)]
pub(crate) struct Cycle {
    #[primary_span]
    pub span: Span,
    pub stack_bottom: String,
    #[subdiagnostic]
    pub cycle_stack: Vec<CycleStack>,
    #[subdiagnostic]
    pub stack_count: StackCount,
    #[subdiagnostic]
    pub alias: Option<Alias>,
    #[subdiagnostic]
    pub cycle_usage: Option<CycleUsage>,
    #[note(
        "see https://rustc-dev-guide.rust-lang.org/overview.html#queries and https://rustc-dev-guide.rust-lang.org/query.html for more information"
    )]
    pub note_span: (),
}

#[derive(Diagnostic)]
#[diag("internal compiler error: reentrant incremental verify failure, suppressing message")]
pub(crate) struct Reentrant;

#[derive(Diagnostic)]
#[diag("internal compiler error: encountered incremental compilation error with {$dep_node}")]
#[note("please follow the instructions below to create a bug report with the provided information")]
#[note("for incremental compilation bugs, having a reproduction is vital")]
#[note(
    "an ideal reproduction consists of the code before and some patch that then triggers the bug when applied and compiled again"
)]
#[note("as a workaround, you can run {$run_cmd} to allow your project to compile")]
pub(crate) struct IncrementCompilation {
    pub run_cmd: String,
    pub dep_node: String,
}

#[derive(Diagnostic)]
#[help(
    "consider increasing the recursion limit by adding a `#![recursion_limit = \"{$suggested_limit}\"]` attribute to your crate (`{$crate_name}`)"
)]
#[diag("queries overflow the depth limit!")]
pub struct QueryOverflow {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub note: QueryOverflowNote,
    pub suggested_limit: Limit,
    pub crate_name: Symbol,
}

#[derive(Subdiagnostic)]
#[note("query depth increased by {$depth} when {$desc}")]
pub struct QueryOverflowNote {
    pub desc: String,
    pub depth: usize,
}
