use rustc_data_structures::fingerprint::Fingerprint;
pub use rustc_lint_defs::AttributeLintKind;
use rustc_lint_defs::LintId;
use rustc_macros::HashStable_Generic;
use rustc_span::Span;

use crate::HirId;

#[derive(Debug)]
pub struct DelayedLints {
    pub lints: Box<[DelayedLint]>,
    // Only present when the crate hash is needed.
    pub opt_hash: Option<Fingerprint>,
}

/// During ast lowering, no lints can be emitted.
/// That is because lints attach to nodes either in the AST, or on the built HIR.
/// When attached to AST nodes, they're emitted just before building HIR,
/// and then there's a gap where no lints can be emitted until HIR is done.
/// The variants in this enum represent lints that are temporarily stashed during
/// AST lowering to be emitted once HIR is built.
#[derive(Debug, HashStable_Generic)]
pub enum DelayedLint {
    AttributeParsing(AttributeLint<HirId>),
}

#[derive(Debug, HashStable_Generic)]
pub struct AttributeLint<Id> {
    pub lint_id: LintId,
    pub id: Id,
    pub span: Span,
    pub kind: AttributeLintKind,
}
