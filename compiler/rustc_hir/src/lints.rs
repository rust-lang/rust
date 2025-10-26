use rustc_data_structures::fingerprint::Fingerprint;
use rustc_macros::HashStable_Generic;
use rustc_span::Span;

use crate::{AttrPath, HirId, Target};

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
#[derive(Clone, Debug, HashStable_Generic)]
pub enum DelayedLint {
    AttributeParsing(AttributeLint<HirId>),
}

#[derive(Clone, Debug, HashStable_Generic)]
pub struct AttributeLint<Id> {
    pub id: Id,
    pub span: Span,
    pub kind: AttributeLintKind,
}

#[derive(Clone, Debug, HashStable_Generic)]
pub enum AttributeLintKind {
    /// Copy of `IllFormedAttributeInput`
    /// specifically for the `invalid_macro_export_arguments` lint until that is removed,
    /// see <https://github.com/rust-lang/rust/pull/143857#issuecomment-3079175663>
    InvalidMacroExportArguments {
        suggestions: Vec<String>,
    },
    UnusedDuplicate {
        this: Span,
        other: Span,
        warning: bool,
    },
    IllFormedAttributeInput {
        suggestions: Vec<String>,
    },
    EmptyAttribute {
        first_span: Span,
        attr_path: AttrPath,
        valid_without_list: bool,
    },
    InvalidTarget {
        name: AttrPath,
        target: Target,
        applied: Vec<String>,
        only: &'static str,
    },
    InvalidStyle {
        name: AttrPath,
        is_used_as_inner: bool,
        target: Target,
        target_span: Span,
    },
}
