//! This pass is overloaded and runs two different lints.
//!
//! - MISSING_DOC_CODE_EXAMPLES: this looks for public items missing doc-tests
//! - PRIVATE_DOC_TESTS: this looks for private items with doc-tests.

use super::{span_of_attrs, Pass};
use crate::clean::*;
use crate::core::DocContext;
use crate::fold::DocFolder;
use crate::html::markdown::{find_testable_code, ErrorCodes, LangString};
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

impl Tests {
    pub(crate) fn new() -> Tests {
        Tests { found_tests: 0 }
    }
}

impl crate::test::Tester for Tests {
    fn add_test(&mut self, _: String, _: LangString, _: usize) {
        self.found_tests += 1;
    }
}

pub fn look_for_tests<'tcx>(cx: &DocContext<'tcx>, dox: &str, item: &Item) {
    let hir_id = match cx.as_local_hir_id(item.def_id) {
        Some(hir_id) => hir_id,
        None => {
            // If non-local, no need to check anything.
            return;
        }
    };

    let mut tests = Tests::new();

    find_testable_code(&dox, &mut tests, ErrorCodes::No, false, None);

    if tests.found_tests == 0 {
        use ItemEnum::*;

        let should_report = match item.inner {
            ExternCrateItem(_, _) | ImportItem(_) | PrimitiveItem(_) | KeywordItem(_) => false,
            _ => true,
        };
        if should_report {
            debug!("reporting error for {:?} (hir_id={:?})", item, hir_id);
            let sp = span_of_attrs(&item.attrs).unwrap_or(item.source.span());
            cx.tcx.struct_span_lint_hir(
                lint::builtin::MISSING_DOC_CODE_EXAMPLES,
                hir_id,
                sp,
                |lint| lint.build("missing code example in this documentation").emit(),
            );
        }
    } else if rustc_feature::UnstableFeatures::from_environment().is_nightly_build()
        && tests.found_tests > 0
        && !cx.renderinfo.borrow().access_levels.is_public(item.def_id)
    {
        cx.tcx.struct_span_lint_hir(
            lint::builtin::PRIVATE_DOC_TESTS,
            hir_id,
            span_of_attrs(&item.attrs).unwrap_or(item.source.span()),
            |lint| lint.build("documentation test in private item").emit(),
        );
    }
}
