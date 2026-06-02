//! Contains information about "passes", used to modify crate information during the documentation
//! process.

use std::alloc::Allocator;

use self::Condition::*;
use crate::clean;
use crate::core::DocContext;

mod stripper;
pub(crate) use stripper::*;

mod strip_aliased_non_local;
pub(crate) use self::strip_aliased_non_local::strip_aliased_non_local_pass;

mod strip_hidden;
pub(crate) use self::strip_hidden::strip_hidden_pass;

mod strip_private;
pub(crate) use self::strip_private::strip_private_pass;

mod strip_priv_imports;
pub(crate) use self::strip_priv_imports::strip_priv_imports_pass;

mod propagate_doc_cfg;
pub(crate) use self::propagate_doc_cfg::propagate_doc_cfg_pass;

mod propagate_stability;
pub(crate) use self::propagate_stability::propagate_stability_pass;

pub(crate) mod collect_intra_doc_links;
pub(crate) use self::collect_intra_doc_links::collect_intra_doc_links_pass;

mod check_doc_test_visibility;
pub(crate) use self::check_doc_test_visibility::check_doc_test_visibility_pass;

mod collect_trait_impls;
pub(crate) use self::collect_trait_impls::collect_trait_impls_pass;

mod calculate_doc_coverage;
pub(crate) use self::calculate_doc_coverage::calculate_doc_coverage_pass;

mod lint;
pub(crate) use self::lint::run_lints_pass;

/// A single pass over the cleaned documentation.
///
/// Runs in the compiler context, so it has access to types and traits and the like.
#[derive(Copy, Clone)]
pub(crate) struct Pass<A: Allocator + Copy> {
    pub(crate) name: &'static str,
    pub(crate) run: Option<fn(clean::Crate, &mut DocContext<'_, A>) -> clean::Crate>,
    pub(crate) description: &'static str,
}

/// In a list of passes, a pass that may or may not need to be run depending on options.
#[derive(Copy, Clone)]
pub(crate) struct ConditionalPass<A: Allocator + Copy> {
    pub(crate) pass: Pass<A>,
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
pub(crate) fn passes<A: Allocator + Copy>() -> [Pass<A>; 11] {
    [
        check_doc_test_visibility_pass(),
        propagate_doc_cfg_pass(),
        strip_aliased_non_local_pass(),
        strip_hidden_pass(),
        strip_private_pass(),
        strip_priv_imports_pass(),
        propagate_stability_pass(),
        collect_intra_doc_links_pass(),
        collect_trait_impls_pass(),
        calculate_doc_coverage_pass(),
        run_lints_pass(),
    ]
}

/// The list of passes run by default.
pub(crate) fn default_passes<A: Allocator + Copy>() -> [ConditionalPass<A>; 10] {
    [
        ConditionalPass::always(collect_trait_impls_pass()),
        ConditionalPass::always(check_doc_test_visibility_pass()),
        ConditionalPass::always(strip_aliased_non_local_pass()),
        ConditionalPass::always(propagate_doc_cfg_pass()),
        ConditionalPass::new(strip_hidden_pass(), WhenNotDocumentHidden),
        ConditionalPass::new(strip_private_pass(), WhenNotDocumentPrivate),
        ConditionalPass::new(strip_priv_imports_pass(), WhenDocumentPrivate),
        ConditionalPass::always(collect_intra_doc_links_pass()),
        ConditionalPass::always(propagate_stability_pass()),
        ConditionalPass::always(run_lints_pass()),
    ]
}

/// The list of default passes run when `--doc-coverage` is passed to rustdoc.
pub(crate) fn coverage_passes<A: Allocator + Copy>() -> [ConditionalPass<A>; 3] {
    [
        ConditionalPass::new(strip_hidden_pass(), WhenNotDocumentHidden),
        ConditionalPass::new(strip_private_pass(), WhenNotDocumentPrivate),
        ConditionalPass::always(calculate_doc_coverage_pass()),
    ]
}

impl<A: Allocator + Copy> ConditionalPass<A> {
    pub(crate) const fn always(pass: Pass<A>) -> Self {
        Self::new(pass, Always)
    }

    pub(crate) const fn new(pass: Pass<A>, condition: Condition) -> Self {
        ConditionalPass { pass, condition }
    }
}

/// Returns the given default set of passes.
pub(crate) fn defaults<A: Allocator + Copy>(show_coverage: bool) -> Box<[ConditionalPass<A>]> {
    if show_coverage { Box::new(coverage_passes()) as Box<[_]> } else { Box::new(default_passes()) }
}
