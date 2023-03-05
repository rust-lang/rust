//! Looks for items missing (or incorrectly having) doctests.
//!
//! This pass is overloaded and runs two different lints.
//!
//! - MISSING_DOC_CODE_EXAMPLES: this lint is **UNSTABLE** and looks for public items missing doctests.
//! - PRIVATE_DOC_TESTS: this lint is **STABLE** and looks for private items with doctests.

use super::Pass;
use crate::clean;
use crate::clean::*;
use crate::core::DocContext;
use crate::html::markdown::{find_testable_code, ErrorCodes, Ignore, LangString};
use crate::visit::DocVisitor;
use crate::visit_ast::inherits_doc_hidden;
use rustc_hir as hir;
use rustc_middle::lint::LintLevelSource;
use rustc_session::lint;

pub(crate) const CHECK_DOC_TEST_VISIBILITY: Pass = Pass {
    name: "check_doc_test_visibility",
    run: check_doc_test_visibility,
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

impl<'a, 'tcx> DocVisitor for DocTestVisibilityLinter<'a, 'tcx> {
    fn visit_item(&mut self, item: &Item) {
        let dox = item.attrs.collapsed_doc_value().unwrap_or_default();

        look_for_tests(self.cx, &dox, item);

        self.visit_item_recur(item)
    }
}

pub(crate) struct Tests {
    pub(crate) found_tests: usize,
}

impl crate::doctest::Tester for Tests {
    fn add_test(&mut self, _: String, config: LangString, _: usize) {
        if config.rust && config.ignore == Ignore::None {
            self.found_tests += 1;
        }
    }
}

pub(crate) fn should_have_doc_example(cx: &DocContext<'_>, item: &clean::Item) -> bool {
    if !cx.cache.effective_visibilities.is_directly_public(cx.tcx, item.item_id.expect_def_id())
        || matches!(
            *item.kind,
            clean::StructFieldItem(_)
                | clean::VariantItem(_)
                | clean::AssocConstItem(..)
                | clean::AssocTypeItem(..)
                | clean::TypedefItem(_)
                | clean::StaticItem(_)
                | clean::ConstantItem(_)
                | clean::ExternCrateItem { .. }
                | clean::ImportItem(_)
                | clean::PrimitiveItem(_)
                | clean::KeywordItem
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
    if let Some(parent_def_id) = cx.tcx.opt_local_parent(def_id) &&
        let Some(parent_node) = cx.tcx.hir().find_by_def_id(parent_def_id) &&
        matches!(
            parent_node,
            hir::Node::Item(hir::Item {
                kind: hir::ItemKind::Impl(hir::Impl { of_trait: Some(_), .. }),
                ..
            })
        )
    {
        return false;
    }

    if cx.tcx.is_doc_hidden(def_id.to_def_id())
        || inherits_doc_hidden(cx.tcx, def_id)
        || cx.tcx.def_span(def_id.to_def_id()).in_derive_expansion()
    {
        return false;
    }
    let (level, source) = cx.tcx.lint_level_at_node(
        crate::lint::MISSING_DOC_CODE_EXAMPLES,
        cx.tcx.hir().local_def_id_to_hir_id(def_id),
    );
    level != lint::Level::Allow || matches!(source, LintLevelSource::Default)
}

pub(crate) fn look_for_tests<'tcx>(cx: &DocContext<'tcx>, dox: &str, item: &Item) {
    let Some(hir_id) = DocContext::as_local_hir_id(cx.tcx, item.item_id)
    else {
        // If non-local, no need to check anything.
        return;
    };

    let mut tests = Tests { found_tests: 0 };

    find_testable_code(dox, &mut tests, ErrorCodes::No, false, None);

    if tests.found_tests == 0 && cx.tcx.features().rustdoc_missing_doc_code_examples {
        if should_have_doc_example(cx, item) {
            debug!("reporting error for {:?} (hir_id={:?})", item, hir_id);
            let sp = item.attr_span(cx.tcx);
            cx.tcx.struct_span_lint_hir(
                crate::lint::MISSING_DOC_CODE_EXAMPLES,
                hir_id,
                sp,
                "missing code example in this documentation",
                |lint| lint,
            );
        }
    } else if tests.found_tests > 0
        && !cx.cache.effective_visibilities.is_exported(cx.tcx, item.item_id.expect_def_id())
    {
        cx.tcx.struct_span_lint_hir(
            crate::lint::PRIVATE_DOC_TESTS,
            hir_id,
            item.attr_span(cx.tcx),
            "documentation test in private item",
            |lint| lint,
        );
    }
}
