use rustc_error_messages::MultiSpan;
use rustc_lint_defs::LintId;
pub use rustc_lint_defs::{AttributeLintKind, FormatWarning};

use crate::HirId;

pub type DelayedLints = Box<[DelayedLint]>;

/// During ast lowering, no lints can be emitted.
/// That is because lints attach to nodes either in the AST, or on the built HIR.
/// When attached to AST nodes, they're emitted just before building HIR,
/// and then there's a gap where no lints can be emitted until HIR is done.
/// The variants in this enum represent lints that are temporarily stashed during
/// AST lowering to be emitted once HIR is built.
#[derive(Debug)]
pub enum DelayedLint {
    AttributeParsing(AttributeLint<HirId>),
}

#[derive(Debug)]
pub struct AttributeLint<Id> {
    pub lint_id: LintId,
    pub id: Id,
    pub span: MultiSpan,
    pub kind: AttributeLintKind,
}
