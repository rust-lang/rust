use rustc_errors::{Diag, DiagDecorator};
use rustc_hir::HirId;
use rustc_resolve::rustdoc::pulldown_cmark::{Event, LinkType, Parser, Tag};
use rustc_resolve::rustdoc::{item_defid_for_markdown_position, source_span_for_markdown_range};

use crate::clean::Item;
use crate::core::DocContext;
use crate::html::markdown::main_body_opts;

pub(crate) fn visit_item(cx: &DocContext<'_>, item: &Item, hir_id: HirId, dox: &str) {
    let Some(span) = item.span(cx.tcx) else { return };
    let mut p = Parser::new_ext(dox, main_body_opts()).into_offset_iter();
    while let Some((event, range)) = p.next() {
        let span = source_span_for_markdown_range(
            cx.tcx,
            &dox,
            &(range.start..range.start + 1),
            &item.attrs.doc_strings,
        )
        .map(|(span, _)| span)
        .unwrap_or(span.inner());
        let item_id_start = item_defid_for_markdown_position(range.start, &item.attrs.doc_strings);
        let item_id_end = item_defid_for_markdown_position(range.end - 1, &item.attrs.doc_strings);
        if item_id_start != item_id_end {
            cx.tcx.emit_node_span_lint(
                crate::lint::UNPORTABLE_MARKDOWN,
                hir_id,
                span,
                DiagDecorator(|lint: &mut Diag<'_, ()>| {
                    lint.primary_message("markdown element starts on one item and ends on another");
                    lint.help("the way this is parsed might change in the future");
                    report_idx(cx, item, dox, range.start, "starts", lint);
                    report_idx(cx, item, dox, range.end - 1, "ends", lint);
                }),
            );
        } else if let Event::Start(Tag::Link {
            link_type:
                LinkType::Reference
                | LinkType::ReferenceUnknown
                | LinkType::Collapsed
                | LinkType::CollapsedUnknown
                | LinkType::Shortcut
                | LinkType::ShortcutUnknown,
            id,
            ..
        }) = event
        {
            if let Some(refdef) = p.reference_definitions().get(&id[..]) {
                let item_id_refdef_start =
                    item_defid_for_markdown_position(refdef.span.start, &item.attrs.doc_strings);
                let item_id_refdef_end =
                    item_defid_for_markdown_position(refdef.span.end - 1, &item.attrs.doc_strings);
                if item_id_refdef_start != item_id_start {
                    cx.tcx.emit_node_span_lint(
                        crate::lint::UNPORTABLE_MARKDOWN,
                        hir_id,
                        span,
                        DiagDecorator(|lint: &mut Diag<'_, ()>| {
                            lint.primary_message(
                                "markdown link and refdef are defined on different items",
                            );
                            lint.help("the way this is parsed might change in the future");
                            report_idx(cx, item, dox, refdef.span.start, "refdef starts", lint);
                            report_idx(cx, item, dox, range.start, "item starts", lint);
                        }),
                    );
                } else if item_id_refdef_end != item_id_start {
                    cx.tcx.emit_node_span_lint(
                        crate::lint::UNPORTABLE_MARKDOWN,
                        hir_id,
                        span,
                        DiagDecorator(|lint: &mut Diag<'_, ()>| {
                            lint.primary_message(
                                "markdown link and refdef are defined on different items",
                            );
                            lint.help("the way this is parsed might change in the future");
                            report_idx(cx, item, dox, refdef.span.end - 1, "refdef ends", lint);
                            report_idx(cx, item, dox, range.start, "item starts", lint);
                        }),
                    );
                }
            }
        }
    }
}

fn report_idx(
    cx: &DocContext<'_>,
    item: &Item,
    dox: &str,
    idx: usize,
    verb: &'static str,
    lint: &mut Diag<'_, ()>,
) {
    if let Some((span, _)) =
        source_span_for_markdown_range(cx.tcx, &dox, &(idx..idx + 1), &item.attrs.doc_strings)
    {
        lint.span_label(span, format!("{verb} here"));
    } else {
        let line_start = dox[..idx].rfind('\n').map_or(0, |i| i + 1);
        let line_end = dox[idx..].find('\n').map_or(dox.len(), |i| i + idx);
        let line = &dox[line_start..line_end];
        lint.help(format!("{verb} near `{line}`"));
    }
}
