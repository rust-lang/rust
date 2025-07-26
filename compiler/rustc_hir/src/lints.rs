use rustc_attr_data_structures::lints::AttributeLint;
use rustc_data_structures::fingerprint::Fingerprint;
use rustc_macros::HashStable_Generic;

use crate::HirId;

/// During ast lowering, no lints can be emitted.
/// That is because lints attach to nodes either in the AST, or on the built HIR.
/// When attached to AST nodes, they're emitted just before building HIR,
/// and then there's a gap where no lints can be emitted until HIR is done.
/// The variants in this enum represent lints that are temporarily stashed during
/// AST lowering to be emitted once HIR is built.
#[derive(Clone, Debug, HashStable_Generic)]
pub enum DelayedLint {
    AttributeParsing(AttributeLint<HirId>),
}

#[derive(Debug)]
pub struct DelayedLints {
    pub lints: Box<[DelayedLint]>,
    // Only present when the crate hash is needed.
    pub opt_hash: Option<Fingerprint>,
}
