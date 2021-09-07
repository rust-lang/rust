//! This pass is overloaded and runs two different lints.
//!
//! - MISSING_DOC_CODE_EXAMPLES: this lint is **UNSTABLE** and looks for public items missing doctests
//! - PRIVATE_DOC_TESTS: this lint is **STABLE** and looks for private items with doctests.

use super::Pass;
use crate::clean;
use crate::clean::*;
use crate::core::DocContext;
use crate::fold::DocFolder;
use crate::html::markdown::{find_testable_code, ErrorCodes, Ignore, LangString};
use crate::visit_ast::inherits_doc_hidden;
use rustc_hir as hir;
use rustc_middle::lint::LintLevelSource;
use rustc_session::lint;
use rustc_span::symbol::sym;

crate const CHECK_PRIVATE_ITEMS_DOC_TESTS: Pass = Pass {
    name: "check-private-items-doc-tests",
    run: check_private_items_doc_tests,
    description: "check private items doc tests",
};

struct PrivateItemDocTestLinter<'a, 'tcx> {
    cx: &'a mut DocContext<'tcx>,
}

crate fn check_private_items_doc_tests(krate: Crate, cx: &mut DocContext<'_>) -> Crate {
    let mut coll = PrivateItemDocTestLinter { cx };

    coll.fold_crate(krate)
}

impl<'a, 'tcx> DocFolder for PrivateItemDocTestLinter<'a, 'tcx> {
    fn fold_item(&mut self, item: Item) -> Option<Item> {
        let dox = item.attrs.collapsed_doc_value().unwrap_or_else(String::new);

        look_for_tests(self.cx, &dox, &item);

        Some(self.fold_item_recur(item))
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

crate fn should_have_doc_example(cx: &DocContext<'_>, item: &clean::Item) -> bool {
    if !cx.cache.access_levels.is_public(item.def_id.expect_def_id())
        || matches!(
            *item.kind,
            clean::StructFieldItem(_)
                | clean::VariantItem(_)
                | clean::AssocConstItem(_, _)
                | clean::AssocTypeItem(_, _)
                | clean::TypedefItem(_, _)
                | clean::StaticItem(_)
                | clean::ConstantItem(_)
                | clean::ExternCrateItem { .. }
                | clean::ImportItem(_)
                | clean::PrimitiveItem(_)
                | clean::KeywordItem(_)
                // check for trait impl
                | clean::ImplItem(clean::Impl { trait_: Some(_), .. })
        )
    {
        return false;
    }

    // The `expect_def_id()` should be okay because `local_def_id_to_hir_id`
    // would presumably panic if a fake `DefIndex` were passed.
    let hir_id = cx.tcx.hir().local_def_id_to_hir_id(item.def_id.expect_def_id().expect_local());

    // check if parent is trait impl
    if let Some(parent_hir_id) = cx.tcx.hir().find_parent_node(hir_id) {
        if let Some(parent_node) = cx.tcx.hir().find(parent_hir_id) {
            if matches!(
                parent_node,
                hir::Node::Item(hir::Item {
                    kind: hir::ItemKind::Impl(hir::Impl { of_trait: Some(_), .. }),
                    ..
                })
            ) {
                return false;
            }
        }
    }

    if cx.tcx.hir().attrs(hir_id).lists(sym::doc).has_word(sym::hidden)
        || inherits_doc_hidden(cx.tcx, hir_id)
        || cx.tcx.hir().span(hir_id).in_derive_expansion()
    {
        return false;
    }
    let (level, source) = cx.tcx.lint_level_at_node(crate::lint::MISSING_DOC_CODE_EXAMPLES, hir_id);
    level != lint::Level::Allow || matches!(source, LintLevelSource::Default)
}

crate fn look_for_tests<'tcx>(cx: &DocContext<'tcx>, dox: &str, item: &Item) {
    let hir_id = match DocContext::as_local_hir_id(cx.tcx, item.def_id) {
        Some(hir_id) => hir_id,
        None => {
            // If non-local, no need to check anything.
            return;
        }
    };

    let mut tests = Tests { found_tests: 0 };

    find_testable_code(&dox, &mut tests, ErrorCodes::No, false, None);

    if tests.found_tests == 0 && cx.tcx.sess.is_nightly_build() {
        if should_have_doc_example(cx, &item) {
            debug!("reporting error for {:?} (hir_id={:?})", item, hir_id);
            let sp = item.attr_span(cx.tcx);
            cx.tcx.struct_span_lint_hir(
                crate::lint::MISSING_DOC_CODE_EXAMPLES,
                hir_id,
                sp,
                |lint| lint.build("missing code example in this documentation").emit(),
            );
        }
    } else if tests.found_tests > 0
        && !cx.cache.access_levels.is_public(item.def_id.expect_def_id())
    {
        cx.tcx.struct_span_lint_hir(
            crate::lint::PRIVATE_DOC_TESTS,
            hir_id,
            item.attr_span(cx.tcx),
            |lint| lint.build("documentation test in private item").emit(),
        );
    }
}
