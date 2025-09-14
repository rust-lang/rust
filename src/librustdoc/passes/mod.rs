//! Contains information about "passes", used to modify crate information during the documentation
//! process.

use self::Condition::*;
use crate::clean;
use crate::core::DocContext;

mod stripper;
pub(crate) use stripper::*;

mod calculate_doc_coverage;
mod check_doc_test_visibility;
pub(crate) mod collect_intra_doc_links;
mod collect_trait_impls;
mod lint;
mod propagate_doc_cfg;
mod propagate_stability;
mod strip_aliased_non_local;
mod strip_hidden;
mod strip_priv_imports;
mod strip_private;

/// A single pass over the cleaned documentation.
///
/// Runs in the compiler context, so it has access to types and traits and the like.
#[derive(Copy, Clone)]
pub(crate) struct Pass {
    pub(crate) name: &'static str,
    pub(crate) run: Option<fn(clean::Crate, &mut DocContext<'_>) -> clean::Crate>,
}

/// In a list of passes, a pass that may or may not need to be run depending on options.
#[derive(Copy, Clone)]
pub(crate) struct ConditionalPass {
    pub(crate) pass: Pass,
    pub(crate) condition: Condition,
}

/// How to decide whether to run a conditional pass.
#[derive(Copy, Clone)]
pub(crate) enum Condition {
    Always,
    /// When `--document-private-items` is passed.
    WhenDocumentPrivate,
    /// When `--document-private-items` is not passed.
    WhenNotDocumentPrivate,
    /// When `--document-hidden-items` is not passed.
    WhenNotDocumentHidden,
}

/// The list of passes run by default.
const DEFAULT_PASSES: &[ConditionalPass] = &[
    ConditionalPass::always(collect_trait_impls::COLLECT_TRAIT_IMPLS),
    ConditionalPass::always(check_doc_test_visibility::CHECK_DOC_TEST_VISIBILITY),
    ConditionalPass::always(strip_aliased_non_local::STRIP_ALIASED_NON_LOCAL),
    ConditionalPass::always(propagate_doc_cfg::PROPAGATE_DOC_CFG),
    ConditionalPass::new(strip_hidden::STRIP_HIDDEN, WhenNotDocumentHidden),
    ConditionalPass::new(strip_private::STRIP_PRIVATE, WhenNotDocumentPrivate),
    ConditionalPass::new(strip_priv_imports::STRIP_PRIV_IMPORTS, WhenDocumentPrivate),
    ConditionalPass::always(collect_intra_doc_links::COLLECT_INTRA_DOC_LINKS),
    ConditionalPass::always(propagate_stability::PROPAGATE_STABILITY),
    ConditionalPass::always(lint::RUN_LINTS),
];

/// The list of default passes run when `--doc-coverage` is passed to rustdoc.
const COVERAGE_PASSES: &[ConditionalPass] = &[
    ConditionalPass::new(strip_hidden::STRIP_HIDDEN, WhenNotDocumentHidden),
    ConditionalPass::new(strip_private::STRIP_PRIVATE, WhenNotDocumentPrivate),
    ConditionalPass::always(calculate_doc_coverage::CALCULATE_DOC_COVERAGE),
];

impl ConditionalPass {
    pub(crate) const fn always(pass: Pass) -> Self {
        Self::new(pass, Always)
    }

    pub(crate) const fn new(pass: Pass, condition: Condition) -> Self {
        ConditionalPass { pass, condition }
    }
}

/// Returns the given default set of passes.
pub(crate) fn defaults(show_coverage: bool) -> &'static [ConditionalPass] {
    if show_coverage { COVERAGE_PASSES } else { DEFAULT_PASSES }
}
