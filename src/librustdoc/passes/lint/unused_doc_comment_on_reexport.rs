//! Detects non-inlined local reexports with doc comments.
use crate::clean::*;
use crate::core::DocContext;

use rustc_errors::Applicability;
use rustc_span::DUMMY_SP;

pub(crate) fn visit_item(cx: &DocContext<'_>, item: &Item) {
    if item.attrs.doc_strings.is_empty() || !matches!(*item.kind, ItemKind::ImportItem(_)) {
        return;
    }
    let Some(hir_id) = DocContext::as_local_hir_id(cx.tcx, item.item_id)
    else {
        // If non-local, no need to check anything.
        return;
    };
    let sp = item.attrs.doc_strings[0].span;
    let item_sp = item.span(cx.tcx).map(|sp| sp.inner()).unwrap_or(DUMMY_SP).shrink_to_lo();

    // If we see an import item with doc, then it's wrong in any case because otherwise, if it
    // had `#[doc(inline)]`, we wouldn't see it here in the first place.
    cx.tcx.struct_span_lint_hir(
        crate::lint::DOCUMENTED_LOCAL_NON_INLINED_REEXPORT,
        hir_id,
        sp,
        "doc comments on local non-inlined reexports are ignored",
        |lint| {
            lint.note(
                "the documentation won't be visible because reexports don't have their own page",
            );
            lint.span_suggestion(
                item_sp,
                "try adding `#[doc(inline)]`",
                "#[doc(inline)] ",
                Applicability::MaybeIncorrect,
            );
            lint.span_suggestion(
                sp,
                "or move the documentation directly on the reexported item",
                "",
                Applicability::MaybeIncorrect,
            );
            lint
        },
    );
}
