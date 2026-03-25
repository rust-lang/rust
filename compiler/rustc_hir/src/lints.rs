use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::sync::{DynSend, DynSync};
use rustc_errors::{Diag, DiagCtxtHandle, Level};
use rustc_lint_defs::LintId;
pub use rustc_lint_defs::{AttributeLintKind, FormatWarning};
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
    AttributeParsing(AttributeLint),
    Dynamic(DynAttribute),
}

#[derive(Debug, HashStable_Generic)]
pub struct AttributeLint {
    pub lint_id: LintId,
    pub id: HirId,
    pub span: Span,
    pub kind: AttributeLintKind,
}

#[derive(HashStable_Generic)]
pub struct DynAttribute {
    pub lint_id: LintId,
    pub id: HirId,
    pub span: Span,
    #[stable_hasher(ignore)]
    pub callback: Box<
        dyn for<'a> Fn(DiagCtxtHandle<'a>, Level) -> Diag<'a, ()> + DynSend + DynSync + 'static,
    >,
}

impl std::fmt::Debug for DynAttribute {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DynAttribute")
            .field("lint_id", &self.lint_id)
            .field("id", &self.id)
            .field("span", &self.span)
            .finish()
    }
}
