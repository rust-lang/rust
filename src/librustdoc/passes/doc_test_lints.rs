//! This pass is overloaded and runs two different lints.
//!
//! - MISSING_DOC_CODE_EXAMPLES: this lint is **UNSTABLE** and looks for public items missing doc-tests
//! - PRIVATE_DOC_TESTS: this lint is **STABLE** and looks for private items with doc-tests.

use super::{span_of_attrs, Pass};
use crate::clean;
use crate::clean::*;
use crate::core::DocContext;
use crate::fold::DocFolder;
use crate::html::markdown::{find_testable_code, ErrorCodes, Ignore, LangString};
use rustc_middle::lint::LintSource;
use rustc_session::lint;

pub const CHECK_PRIVATE_ITEMS_DOC_TESTS: Pass = Pass {
    name: "check-private-items-doc-tests",
    run: check_private_items_doc_tests,
    description: "check private items doc tests",
};

struct PrivateItemDocTestLinter<'a, 'tcx> {
    cx: &'a DocContext<'tcx>,
}

impl<'a, 'tcx> PrivateItemDocTestLinter<'a, 'tcx> {
    fn new(cx: &'a DocContext<'tcx>) -> Self {
        PrivateItemDocTestLinter { cx }
    }
}

pub fn check_private_items_doc_tests(krate: Crate, cx: &DocContext<'_>) -> Crate {
    let mut coll = PrivateItemDocTestLinter::new(cx);

    coll.fold_crate(krate)
}

impl<'a, 'tcx> DocFolder for PrivateItemDocTestLinter<'a, 'tcx> {
    fn fold_item(&mut self, item: Item) -> Option<Item> {
        let cx = self.cx;
        let dox = item.attrs.collapsed_doc_value().unwrap_or_else(String::new);

        look_for_tests(&cx, &dox, &item);

        self.fold_item_recur(item)
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

pub fn should_have_doc_example(cx: &DocContext<'_>, item: &clean::Item) -> bool {
    if matches!(item.kind,
        clean::StructFieldItem(_)
        | clean::VariantItem(_)
        | clean::AssocConstItem(_, _)
        | clean::AssocTypeItem(_, _)
        | clean::TypedefItem(_, _)
        | clean::StaticItem(_)
        | clean::ConstantItem(_)
        | clean::ExternCrateItem(_, _)
        | clean::ImportItem(_)
        | clean::PrimitiveItem(_)
        | clean::KeywordItem(_)
    ) {
        return false;
    }
    let hir_id = cx.tcx.hir().local_def_id_to_hir_id(item.def_id.expect_local());
    let (level, source) =
        cx.tcx.lint_level_at_node(lint::builtin::MISSING_DOC_CODE_EXAMPLES, hir_id);
    level != lint::Level::Allow || matches!(source, LintSource::Default)
}

pub fn look_for_tests<'tcx>(cx: &DocContext<'tcx>, dox: &str, item: &Item) {
    let hir_id = match cx.as_local_hir_id(item.def_id) {
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
            let sp = span_of_attrs(&item.attrs).unwrap_or(item.source.span());
            cx.tcx.struct_span_lint_hir(
                lint::builtin::MISSING_DOC_CODE_EXAMPLES,
                hir_id,
                sp,
                |lint| lint.build("missing code example in this documentation").emit(),
            );
        }
    } else if tests.found_tests > 0 && !cx.renderinfo.borrow().access_levels.is_public(item.def_id)
    {
        cx.tcx.struct_span_lint_hir(
            lint::builtin::PRIVATE_DOC_TESTS,
            hir_id,
            span_of_attrs(&item.attrs).unwrap_or(item.source.span()),
            |lint| lint.build("documentation test in private item").emit(),
        );
    }
}
