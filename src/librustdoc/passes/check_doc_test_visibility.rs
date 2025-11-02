//! Looks for items missing (or incorrectly having) doctests.
//!
//! This pass is overloaded and runs two different lints.
//!
//! - MISSING_DOC_CODE_EXAMPLES: this lint is **UNSTABLE** and looks for public items missing doctests.
//! - PRIVATE_DOC_TESTS: this lint is **STABLE** and looks for private items with doctests.

use rustc_hir as hir;
use rustc_middle::lint::{LevelAndSource, LintLevelSource};
use rustc_session::lint;
use tracing::debug;

use super::Pass;
use crate::clean;
use crate::clean::utils::inherits_doc_hidden;
use crate::clean::*;
use crate::core::DocContext;
use crate::html::markdown::{ErrorCodes, Ignore, LangString, MdRelLine, find_testable_code};
use crate::visit::DocVisitor;

pub(crate) const CHECK_DOC_TEST_VISIBILITY: Pass = Pass {
    name: "check_doc_test_visibility",
    run: Some(check_doc_test_visibility),
    description: "run various visibility-related lints on doctests",
};

struct DocTestVisibilityLinter<'a, 'tcx> {
    cx: &'a mut DocContext<'tcx>,
}

pub(crate) fn check_doc_test_visibility(krate: Crate, cx: &mut DocContext<'_>) -> Crate {
    let mut coll = DocTestVisibilityLinter { cx };
    coll.visit_crate(&krate);
    krate
}

impl DocVisitor<'_> for DocTestVisibilityLinter<'_, '_> {
    fn visit_item(&mut self, item: &Item) {
        look_for_tests(self.cx, &item.doc_value(), item);

        self.visit_item_recur(item)
    }
}

pub(crate) struct Tests {
    pub(crate) found_tests: usize,
}

impl crate::doctest::DocTestVisitor for Tests {
    fn visit_test(&mut self, _: String, config: LangString, _: MdRelLine) {
        if config.rust && config.ignore == Ignore::None {
            self.found_tests += 1;
        }
    }
}

pub(crate) fn should_have_doc_example(cx: &DocContext<'_>, item: &clean::Item) -> bool {
    if !cx.cache.effective_visibilities.is_directly_public(cx.tcx, item.item_id.expect_def_id())
        || matches!(
            item.kind,
            clean::StructFieldItem(_)
                | clean::VariantItem(_)
                | clean::TypeAliasItem(_)
                | clean::StaticItem(_)
                | clean::ConstantItem(..)
                | clean::ExternCrateItem { .. }
                | clean::ImportItem(_)
                | clean::PrimitiveItem(_)
                | clean::KeywordItem
                | clean::AttributeItem
                | clean::ModuleItem(_)
                | clean::TraitAliasItem(_)
                | clean::ForeignFunctionItem(..)
                | clean::ForeignStaticItem(..)
                | clean::ForeignTypeItem
                | clean::AssocTypeItem(..)
                | clean::RequiredAssocConstItem(..)
                | clean::ProvidedAssocConstItem(..)
                | clean::ImplAssocConstItem(..)
                | clean::RequiredAssocTypeItem(..)
                // check for trait impl
                | clean::ImplItem(box clean::Impl { trait_: Some(_), .. })
        )
    {
        return false;
    }

    // The `expect_def_id()` should be okay because `local_def_id_to_hir_id`
    // would presumably panic if a fake `DefIndex` were passed.
    let def_id = item.item_id.expect_def_id().expect_local();

    // check if parent is trait impl
    if let Some(parent_def_id) = cx.tcx.opt_local_parent(def_id)
        && matches!(
            cx.tcx.hir_node_by_def_id(parent_def_id),
            hir::Node::Item(hir::Item {
                kind: hir::ItemKind::Impl(hir::Impl { of_trait: Some(_), .. }),
                ..
            })
        )
    {
        return false;
    }

    if (!cx.render_options.document_hidden
        && (cx.tcx.is_doc_hidden(def_id.to_def_id()) || inherits_doc_hidden(cx.tcx, def_id, None)))
        || cx.tcx.def_span(def_id.to_def_id()).in_derive_expansion()
    {
        return false;
    }
    let LevelAndSource { level, src, .. } = cx.tcx.lint_level_at_node(
        crate::lint::MISSING_DOC_CODE_EXAMPLES,
        cx.tcx.local_def_id_to_hir_id(def_id),
    );
    level != lint::Level::Allow || matches!(src, LintLevelSource::Default)
}

pub(crate) fn look_for_tests(cx: &DocContext<'_>, dox: &str, item: &Item) {
    let Some(hir_id) = DocContext::as_local_hir_id(cx.tcx, item.item_id) else {
        // If non-local, no need to check anything.
        return;
    };

    let mut tests = Tests { found_tests: 0 };

    find_testable_code(dox, &mut tests, ErrorCodes::No, None);

    if tests.found_tests == 0 && cx.tcx.features().rustdoc_missing_doc_code_examples() {
        if should_have_doc_example(cx, item) {
            debug!("reporting error for {item:?} (hir_id={hir_id:?})");
            let sp = item.attr_span(cx.tcx);
            cx.tcx.node_span_lint(crate::lint::MISSING_DOC_CODE_EXAMPLES, hir_id, sp, |lint| {
                lint.primary_message("missing code example in this documentation");
            });
        }
    } else if tests.found_tests > 0
        && !cx.cache.effective_visibilities.is_exported(cx.tcx, item.item_id.expect_def_id())
    {
        cx.tcx.node_span_lint(
            crate::lint::PRIVATE_DOC_TESTS,
            hir_id,
            item.attr_span(cx.tcx),
            |lint| {
                lint.primary_message("documentation test in private item");
            },
        );
    }
}
