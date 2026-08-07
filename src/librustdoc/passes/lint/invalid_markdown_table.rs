//! Detects table rows where some content seems to have been discarded because there are too many
//! pipe characters.

use std::ops::Range;

use rustc_hir::HirId;
use rustc_macros::Diagnostic;
use rustc_resolve::rustdoc::pulldown_cmark::{Event, Parser, Tag, TagEnd};
use rustc_resolve::rustdoc::source_span_for_markdown_range;

use crate::clean::*;
use crate::core::DocContext;
use crate::html::markdown::main_body_opts;

#[derive(Diagnostic)]
#[diag("table row has too many columns")]
#[help(r"to escape `|` characters in tables, add a `\` before them like `\|`")]
struct UnescapedPipeInTableCell {
    #[primary_span]
    #[label("any content after this column divider is discarded")]
    span: rustc_span::Span,
}

#[derive(Diagnostic)]
#[diag("unused content after last table cell")]
struct ContentAfterLastPipe {
    #[primary_span]
    #[label("this content is discarded")]
    span: rustc_span::Span,
}

pub(crate) fn visit_item(cx: &DocContext<'_>, item: &Item, hir_id: HirId, dox: &str) {
    let mut p = Parser::new_ext(dox, main_body_opts()).into_offset_iter();

    while let Some((event, _range)) = p.next() {
        if Event::Start(Tag::TableRow) == event {
            let mut prev_range = None;
            while let Some((event, range)) = p.next() {
                match event {
                    Event::End(TagEnd::TableCell) => {
                        prev_range = Some(range);
                    }
                    Event::End(TagEnd::TableRow) => {
                        if let Some(prev_range) = &prev_range
                            // So here what is happening: when `pulldown-cmark` is parsing a table
                            // and a table row has too many cells, it doesn't emit events for the
                            // extra cells. So the only way for us to know these extra cells exist
                            // is to compare the row's span with the last emitted cell event's span.
                            // If the span ends don't match, then there are extra cells.
                            && prev_range.end + 1 < range.end
                        {
                            // Something seems wrong, the range diff doesn't match, some content
                            // was left out.
                            let mut after_last_cell_range =
                                Range { start: prev_range.end + 1, end: range.end };
                            if dox[after_last_cell_range.clone()].trim().is_empty() {
                                // Seems all good so let's ignore it and continue;.
                                continue;
                            }
                            // Check if any pipes appear after the end of the row.
                            let mut iter = dox[after_last_cell_range.clone()].bytes().peekable();
                            let mut found_divider = false;
                            while let Some(c) = iter.next() {
                                // the sequence `\\|` still escapes the pipe because GFM
                                // processes block structures like tables in its own pass
                                if c == b'\\' && iter.peek() == Some(&b'|') {
                                    iter.next();
                                } else if c == b'|' {
                                    found_divider = true;
                                    break;
                                }
                            }
                            if found_divider {
                                // Seems like a pipe was not escaped as it should have been.
                                let last_cell_separator =
                                    Range { start: prev_range.end, end: prev_range.end + 1 };

                                if let Some((span, _)) = source_span_for_markdown_range(
                                    cx.tcx,
                                    dox,
                                    &last_cell_separator,
                                    &item.attrs.doc_strings,
                                ) {
                                    cx.tcx.emit_node_span_lint(
                                        crate::lint::INVALID_MARKDOWN_TABLE,
                                        hir_id,
                                        span,
                                        UnescapedPipeInTableCell { span },
                                    );
                                }
                            } else {
                                // An unclosed cell maybe? There is content after the last cell so
                                // let's lint about it.
                                let content = &dox[after_last_cell_range.clone()];
                                after_last_cell_range.end -=
                                    content.len() - content.trim_end().len();

                                if let Some((span, _)) = source_span_for_markdown_range(
                                    cx.tcx,
                                    dox,
                                    &after_last_cell_range,
                                    &item.attrs.doc_strings,
                                ) {
                                    cx.tcx.emit_node_span_lint(
                                        crate::lint::INVALID_MARKDOWN_TABLE,
                                        hir_id,
                                        span,
                                        ContentAfterLastPipe { span },
                                    );
                                }
                            }
                        }
                    }
                    Event::End(TagEnd::Table) => break,
                    _ => {}
                }
            }
        }
    }
}
