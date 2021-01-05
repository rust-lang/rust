//! NIGHTLY & UNSTABLE CHECK: custom_code_classes_in_docs
//!
//! This pass will produce errors when finding custom classes outside of
//! nightly + relevant feature active.

use super::{span_of_attrs, Pass};
use crate::clean::{Crate, Item};
use crate::core::DocContext;
use crate::fold::DocFolder;
use crate::html::markdown::{find_testable_code, ErrorCodes, LangString};

use rustc_session::parse::feature_err;
use rustc_span::symbol::sym;

crate const CHECK_CUSTOM_CODE_CLASSES: Pass = Pass {
    name: "check-custom-code-classes",
    run: check_custom_code_classes,
    description: "check for custom code classes while not in nightly",
};

crate fn check_custom_code_classes(krate: Crate, cx: &mut DocContext<'_>) -> Crate {
    let mut coll = CustomCodeClassLinter { cx };

    coll.fold_crate(krate)
}

struct CustomCodeClassLinter<'a, 'tcx> {
    cx: &'a DocContext<'tcx>,
}

impl<'a, 'tcx> DocFolder for CustomCodeClassLinter<'a, 'tcx> {
    fn fold_item(&mut self, item: Item) -> Option<Item> {
        let dox = item.attrs.collapsed_doc_value().unwrap_or_default();

        look_for_custom_classes(&self.cx, &dox, &item);

        Some(self.fold_item_recur(item))
    }
}

struct TestsWithCustomClasses {
    custom_classes_found: Vec<String>,
}

impl crate::doctest::Tester for TestsWithCustomClasses {
    fn add_test(&mut self, _: String, config: LangString, _: usize) {
        self.custom_classes_found.extend(config.added_classes.into_iter());
    }
}

crate fn look_for_custom_classes<'tcx>(cx: &DocContext<'tcx>, dox: &str, item: &Item) {
    let hir_id = match DocContext::as_local_hir_id(cx.tcx, item.def_id) {
        Some(hir_id) => hir_id,
        None => {
            // If non-local, no need to check anything.
            return;
        }
    };

    let mut tests = TestsWithCustomClasses { custom_classes_found: vec![] };

    find_testable_code(&dox, &mut tests, ErrorCodes::No, false, None);

    if !tests.custom_classes_found.is_empty() && !cx.tcx.features().custom_code_classes_in_docs {
        debug!("reporting error for {:?} (hid_id={:?})", item, hir_id);
        let sp = span_of_attrs(&item.attrs).unwrap_or(item.source.span());
        feature_err(
            &cx.tcx.sess.parse_sess,
            sym::custom_code_classes_in_docs,
            sp,
            "custom classes in code blocks are unstable",
        )
        .note(
            // This will list the wrong items to make them more easily searchable.
            // To ensure the most correct hits, it adds back the 'class:' that was stripped.
            &format!(
                "found these custom classes: class:{}",
                tests.custom_classes_found.join(", class:")
            ),
        )
        .emit();
    }
}
