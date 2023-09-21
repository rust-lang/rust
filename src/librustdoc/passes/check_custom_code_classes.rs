//! NIGHTLY & UNSTABLE CHECK: custom_code_classes_in_docs
//!
//! This pass will produce errors when finding custom classes outside of
//! nightly + relevant feature active.

use super::Pass;
use crate::clean::{Crate, Item};
use crate::core::DocContext;
use crate::fold::DocFolder;
use crate::html::markdown::{find_codes, ErrorCodes, LangString};

use rustc_errors::StashKey;
use rustc_feature::GateIssue;
use rustc_session::parse::add_feature_diagnostics_for_issue;
use rustc_span::symbol::sym;

pub(crate) const CHECK_CUSTOM_CODE_CLASSES: Pass = Pass {
    name: "check-custom-code-classes",
    run: check_custom_code_classes,
    description: "check for custom code classes without the feature-gate enabled",
};

pub(crate) fn check_custom_code_classes(krate: Crate, cx: &mut DocContext<'_>) -> Crate {
    if cx.tcx.features().custom_code_classes_in_docs {
        // Nothing to check here if the feature is enabled.
        return krate;
    }
    let mut coll = CustomCodeClassLinter { cx };

    coll.fold_crate(krate)
}

struct CustomCodeClassLinter<'a, 'tcx> {
    cx: &'a DocContext<'tcx>,
}

impl<'a, 'tcx> DocFolder for CustomCodeClassLinter<'a, 'tcx> {
    fn fold_item(&mut self, item: Item) -> Option<Item> {
        look_for_custom_classes(&self.cx, &item);
        Some(self.fold_item_recur(item))
    }
}

#[derive(Debug)]
struct TestsWithCustomClasses {
    custom_classes_found: Vec<String>,
}

impl crate::doctest::Tester for TestsWithCustomClasses {
    fn add_test(&mut self, _: String, config: LangString, _: usize) {
        self.custom_classes_found.extend(config.added_classes.into_iter());
    }
}

pub(crate) fn look_for_custom_classes<'tcx>(cx: &DocContext<'tcx>, item: &Item) {
    if !item.item_id.is_local() {
        // If non-local, no need to check anything.
        return;
    }

    let mut tests = TestsWithCustomClasses { custom_classes_found: vec![] };

    let dox = item.attrs.doc_value();
    find_codes(&dox, &mut tests, ErrorCodes::No, false, None, true, true);

    if !tests.custom_classes_found.is_empty() {
        let span = item.attr_span(cx.tcx);
        let sess = &cx.tcx.sess.parse_sess;
        let mut err = sess
            .span_diagnostic
            .struct_span_warn(span, "custom classes in code blocks will change behaviour");
        add_feature_diagnostics_for_issue(
            &mut err,
            sess,
            sym::custom_code_classes_in_docs,
            GateIssue::Language,
            false,
        );

        err.note(
            // This will list the wrong items to make them more easily searchable.
            // To ensure the most correct hits, it adds back the 'class:' that was stripped.
            format!(
                "found these custom classes: class={}",
                tests.custom_classes_found.join(",class=")
            ),
        );

        // A later feature_err call can steal and cancel this warning.
        err.stash(span, StashKey::EarlySyntaxWarning);
    }
}
