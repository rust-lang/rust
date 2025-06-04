//! Contains information about "passes", used to modify crate information during the documentation
//! process.

use self::Condition::*;
use crate::clean;
use crate::core::DocContext;

mod stripper;
pub(crate) use stripper::*;

mod strip_aliased_non_local;
pub(crate) use self::strip_aliased_non_local::STRIP_ALIASED_NON_LOCAL;

mod strip_hidden;
pub(crate) use self::strip_hidden::STRIP_HIDDEN;

mod strip_private;
pub(crate) use self::strip_private::STRIP_PRIVATE;

mod strip_priv_imports;
pub(crate) use self::strip_priv_imports::STRIP_PRIV_IMPORTS;

mod propagate_doc_cfg;
pub(crate) use self::propagate_doc_cfg::PROPAGATE_DOC_CFG;

mod propagate_stability;
pub(crate) use self::propagate_stability::PROPAGATE_STABILITY;

pub(crate) mod collect_intra_doc_links;
pub(crate) use self::collect_intra_doc_links::COLLECT_INTRA_DOC_LINKS;

mod check_doc_test_visibility;
pub(crate) use self::check_doc_test_visibility::CHECK_DOC_TEST_VISIBILITY;

mod check_doc_cfg;
pub(crate) use self::check_doc_cfg::CHECK_DOC_CFG;

mod collect_trait_impls;
pub(crate) use self::collect_trait_impls::COLLECT_TRAIT_IMPLS;

mod calculate_doc_coverage;
pub(crate) use self::calculate_doc_coverage::CALCULATE_DOC_COVERAGE;

mod lint;
pub(crate) use self::lint::RUN_LINTS;

/// A single pass over the cleaned documentation.
///
/// Runs in the compiler context, so it has access to types and traits and the like.
#[derive(Copy, Clone)]
pub(crate) struct Pass {
    pub(crate) name: &'static str,
    pub(crate) run: Option<fn(clean::Crate, &mut DocContext<'_>) -> clean::Crate>,
    pub(crate) description: &'static str,
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

/// The full list of passes.
pub(crate) const PASSES: &[Pass] = &[
    CHECK_DOC_CFG,
    CHECK_DOC_TEST_VISIBILITY,
    STRIP_ALIASED_NON_LOCAL,
    STRIP_HIDDEN,
    STRIP_PRIVATE,
    STRIP_PRIV_IMPORTS,
    PROPAGATE_DOC_CFG,
    PROPAGATE_STABILITY,
    COLLECT_INTRA_DOC_LINKS,
    COLLECT_TRAIT_IMPLS,
    CALCULATE_DOC_COVERAGE,
    RUN_LINTS,
];

/// The list of passes run by default.
pub(crate) const DEFAULT_PASSES: &[ConditionalPass] = &[
    ConditionalPass::always(COLLECT_TRAIT_IMPLS),
    ConditionalPass::always(CHECK_DOC_TEST_VISIBILITY),
    ConditionalPass::always(CHECK_DOC_CFG),
    ConditionalPass::always(STRIP_ALIASED_NON_LOCAL),
    ConditionalPass::new(STRIP_HIDDEN, WhenNotDocumentHidden),
    ConditionalPass::new(STRIP_PRIVATE, WhenNotDocumentPrivate),
    ConditionalPass::new(STRIP_PRIV_IMPORTS, WhenDocumentPrivate),
    ConditionalPass::always(COLLECT_INTRA_DOC_LINKS),
    ConditionalPass::always(PROPAGATE_DOC_CFG),
    ConditionalPass::always(PROPAGATE_STABILITY),
    ConditionalPass::always(RUN_LINTS),
];

/// The list of default passes run when `--doc-coverage` is passed to rustdoc.
pub(crate) const COVERAGE_PASSES: &[ConditionalPass] = &[
    ConditionalPass::new(STRIP_HIDDEN, WhenNotDocumentHidden),
    ConditionalPass::new(STRIP_PRIVATE, WhenNotDocumentPrivate),
    ConditionalPass::always(CALCULATE_DOC_COVERAGE),
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
