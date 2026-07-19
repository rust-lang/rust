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
#[help("to escape `|` characters in tables, add a `\\` before them like `\\|`")]
struct UnescapedPipeInTableCell {
    #[primary_span]
    #[label("any content after this column divider is discarded")]
    span: rustc_span::Span,
}

pub(crate) fn visit_item(cx: &DocContext<'_>, item: &Item, hir_id: HirId, dox: &str) {
    let mut p = Parser::new_ext(dox, main_body_opts()).into_offset_iter();

    while let Some((event, _range)) = p.next() {
        if let Event::Start(Tag::Table(_)) = event
            && let Some((Event::Start(Tag::TableHead), _)) = p.next()
        {
            let mut expected_cells = 0;
            while let Some((event, _)) = p.next() {
                match event {
                    Event::End(TagEnd::TableCell) => expected_cells += 1,
                    Event::End(TagEnd::TableHead) => break,
                    _ => {}
                }
            }
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
                            && prev_range.end + 1 != range.end
                        {
                            // Something seems wrong, the range diff doesn't match, some content
                            // was left out. We now check the number of unescaped `|`.
                            let row = &dox[range.clone()];
                            let mut iter = row.chars();
                            let mut divider_count = 0;
                            while let Some(c) = iter.next() {
                                if c == '\\' {
                                    iter.next();
                                } else if c == '|' {
                                    divider_count += 1;
                                }
                            }
                            // + 1 is to handle the `|` at the end of the table row.
                            if divider_count <= expected_cells + 1
                                || dox[Range { start: prev_range.end + 1, end: range.end }]
                                    .trim()
                                    .is_empty()
                            {
                                // Seems all good so let's ignore it and continue;.
                                continue;
                            }
                            let last_cell_separator =
                                Range { start: prev_range.end, end: prev_range.end + 1 };

                            if let Some((span, _)) = source_span_for_markdown_range(
                                cx.tcx,
                                dox,
                                &last_cell_separator,
                                &item.attrs.doc_strings,
                            ) {
                                cx.tcx.emit_node_span_lint(
                                    crate::lint::UNESCAPED_PIPE_IN_TABLE_CELL,
                                    hir_id,
                                    span,
                                    UnescapedPipeInTableCell { span },
                                );
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
