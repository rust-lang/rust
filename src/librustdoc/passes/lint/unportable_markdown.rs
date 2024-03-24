//! Detects markdown syntax that's different between pulldown-cmark
//! 0.9 and 0.10.

use crate::clean::Item;
use crate::core::DocContext;
use crate::html::markdown::main_body_opts;
use pulldown_cmark as cmarko;
use pulldown_cmark_new as cmarkn;
use rustc_resolve::rustdoc::source_span_for_markdown_range;

pub(crate) fn visit_item(cx: &DocContext<'_>, item: &Item) {
    let tcx = cx.tcx;
    let Some(hir_id) = DocContext::as_local_hir_id(tcx, item.item_id) else {
        // If non-local, no need to check anything.
        return;
    };

    let dox = item.doc_value();
    if dox.is_empty() {
        return;
    }

    let link_names = item.link_names(&cx.cache);
    let mut replacer_old = |broken_link: cmarko::BrokenLink<'_>| {
        link_names
            .iter()
            .find(|link| *link.original_text == *broken_link.reference)
            .map(|link| ((*link.href).into(), (*link.new_text).into()))
    };
    let parser_old = cmarko::Parser::new_with_broken_link_callback(
        &dox,
        main_body_opts(),
        Some(&mut replacer_old),
    )
    .into_offset_iter()
    // Not worth cleaning up minor "distinctions without difference" in the AST.
    // Text events get chopped up differently between versions.
    // <html> and `code` mistakes are usually covered by unescaped_backticks and html_tags lints.
    .filter(|(event, _event_range)| {
        !matches!(
            event,
            cmarko::Event::Code(_)
                | cmarko::Event::Text(_)
                | cmarko::Event::Html(_)
                | cmarko::Event::SoftBreak
        )
    });

    pub fn main_body_opts_new() -> cmarkn::Options {
        cmarkn::Options::ENABLE_TABLES
            | cmarkn::Options::ENABLE_FOOTNOTES
            | cmarkn::Options::ENABLE_STRIKETHROUGH
            | cmarkn::Options::ENABLE_TASKLISTS
            | cmarkn::Options::ENABLE_SMART_PUNCTUATION
    }
    let mut replacer_new = |broken_link: cmarkn::BrokenLink<'_>| {
        link_names
            .iter()
            .find(|link| *link.original_text.trim() == *broken_link.reference.trim())
            .map(|link| ((*link.href).into(), (*link.new_text).into()))
    };
    let parser_new = cmarkn::Parser::new_with_broken_link_callback(
        &dox,
        main_body_opts_new(),
        Some(&mut replacer_new),
    )
    .into_offset_iter()
    .filter(|(event, _event_range)| {
        !matches!(
            event,
            cmarkn::Event::Code(_)
                | cmarkn::Event::Text(_)
                | cmarkn::Event::Html(_)
                | cmarkn::Event::InlineHtml(_)
                | cmarkn::Event::Start(cmarkn::Tag::HtmlBlock)
                | cmarkn::Event::End(cmarkn::TagEnd::HtmlBlock)
                | cmarkn::Event::SoftBreak
        )
    });

    let mut reported_an_error = false;
    for ((event_old, event_range_old), (event_new, event_range_new)) in parser_old.zip(parser_new) {
        match (event_old, event_new) {
            (
                cmarko::Event::Start(cmarko::Tag::Emphasis),
                cmarkn::Event::Start(cmarkn::Tag::Emphasis),
            )
            | (
                cmarko::Event::Start(cmarko::Tag::Strong),
                cmarkn::Event::Start(cmarkn::Tag::Strong),
            )
            | (
                cmarko::Event::Start(cmarko::Tag::Strikethrough),
                cmarkn::Event::Start(cmarkn::Tag::Strikethrough),
            )
            | (
                cmarko::Event::End(cmarko::Tag::Emphasis),
                cmarkn::Event::End(cmarkn::TagEnd::Emphasis),
            )
            | (
                cmarko::Event::End(cmarko::Tag::Strong),
                cmarkn::Event::End(cmarkn::TagEnd::Strong),
            )
            | (
                cmarko::Event::End(cmarko::Tag::Strikethrough),
                cmarkn::Event::End(cmarkn::TagEnd::Strikethrough),
            )
            | (
                cmarko::Event::End(cmarko::Tag::Link(..)),
                cmarkn::Event::End(cmarkn::TagEnd::Link),
            )
            | (
                cmarko::Event::End(cmarko::Tag::Image(..)),
                cmarkn::Event::End(cmarkn::TagEnd::Image),
            )
            | (cmarko::Event::FootnoteReference(..), cmarkn::Event::FootnoteReference(..))
            | (cmarko::Event::TaskListMarker(false), cmarkn::Event::TaskListMarker(false))
            | (cmarko::Event::TaskListMarker(true), cmarkn::Event::TaskListMarker(true))
                if event_range_old == event_range_new =>
            {
                // Matching tags. Do nothing.
            }
            (
                cmarko::Event::Start(cmarko::Tag::Link(_, old_dest_url, old_title)),
                cmarkn::Event::Start(cmarkn::Tag::Link { dest_url, title, .. }),
            )
            | (
                cmarko::Event::Start(cmarko::Tag::Image(_, old_dest_url, old_title)),
                cmarkn::Event::Start(cmarkn::Tag::Image { dest_url, title, .. }),
            ) if event_range_old == event_range_new
                && &old_dest_url[..] == &dest_url[..]
                && &old_title[..] == &title[..] =>
            {
                // Matching tags. Do nothing.
            }
            (cmarko::Event::SoftBreak, cmarkn::Event::SoftBreak)
            | (cmarko::Event::HardBreak, cmarkn::Event::HardBreak)
            | (cmarko::Event::Rule, cmarkn::Event::Rule)
            | (
                cmarko::Event::Start(cmarko::Tag::Paragraph),
                cmarkn::Event::Start(cmarkn::Tag::Paragraph),
            )
            | (
                cmarko::Event::Start(cmarko::Tag::Heading(..)),
                cmarkn::Event::Start(cmarkn::Tag::Heading { .. }),
            )
            | (
                cmarko::Event::Start(cmarko::Tag::BlockQuote),
                cmarkn::Event::Start(cmarkn::Tag::BlockQuote),
            )
            | (
                cmarko::Event::Start(cmarko::Tag::CodeBlock(..)),
                cmarkn::Event::Start(cmarkn::Tag::CodeBlock(..)),
            )
            | (
                cmarko::Event::Start(cmarko::Tag::List(..)),
                cmarkn::Event::Start(cmarkn::Tag::List(..)),
            )
            | (cmarko::Event::Start(cmarko::Tag::Item), cmarkn::Event::Start(cmarkn::Tag::Item))
            | (
                cmarko::Event::Start(cmarko::Tag::FootnoteDefinition(..)),
                cmarkn::Event::Start(cmarkn::Tag::FootnoteDefinition(..)),
            )
            | (
                cmarko::Event::Start(cmarko::Tag::Table(..)),
                cmarkn::Event::Start(cmarkn::Tag::Table(..)),
            )
            | (
                cmarko::Event::Start(cmarko::Tag::TableHead),
                cmarkn::Event::Start(cmarkn::Tag::TableHead),
            )
            | (
                cmarko::Event::Start(cmarko::Tag::TableRow),
                cmarkn::Event::Start(cmarkn::Tag::TableRow),
            )
            | (
                cmarko::Event::Start(cmarko::Tag::TableCell),
                cmarkn::Event::Start(cmarkn::Tag::TableCell),
            )
            | (
                cmarko::Event::End(cmarko::Tag::Paragraph),
                cmarkn::Event::End(cmarkn::TagEnd::Paragraph),
            )
            | (
                cmarko::Event::End(cmarko::Tag::Heading(..)),
                cmarkn::Event::End(cmarkn::TagEnd::Heading(_)),
            )
            | (
                cmarko::Event::End(cmarko::Tag::BlockQuote),
                cmarkn::Event::End(cmarkn::TagEnd::BlockQuote),
            )
            | (
                cmarko::Event::End(cmarko::Tag::CodeBlock(..)),
                cmarkn::Event::End(cmarkn::TagEnd::CodeBlock),
            )
            | (
                cmarko::Event::End(cmarko::Tag::List(..)),
                cmarkn::Event::End(cmarkn::TagEnd::List(_)),
            )
            | (cmarko::Event::End(cmarko::Tag::Item), cmarkn::Event::End(cmarkn::TagEnd::Item))
            | (
                cmarko::Event::End(cmarko::Tag::FootnoteDefinition(..)),
                cmarkn::Event::End(cmarkn::TagEnd::FootnoteDefinition),
            )
            | (
                cmarko::Event::End(cmarko::Tag::Table(..)),
                cmarkn::Event::End(cmarkn::TagEnd::Table),
            )
            | (
                cmarko::Event::End(cmarko::Tag::TableHead),
                cmarkn::Event::End(cmarkn::TagEnd::TableHead),
            )
            | (
                cmarko::Event::End(cmarko::Tag::TableRow),
                cmarkn::Event::End(cmarkn::TagEnd::TableRow),
            )
            | (
                cmarko::Event::End(cmarko::Tag::TableCell),
                cmarkn::Event::End(cmarkn::TagEnd::TableCell),
            ) => {
                // Matching tags. Do nothing.
                //
                // Parsers sometimes differ in what they consider the "range of an event,"
                // even though the event is really the same. Inlines are pretty consistent,
                // but stuff like list items? Not really.
                //
                // Mismatched block elements will usually nest differently, so ignoring it
                // works good enough.
            }
            // If we've already reported an error on the start tag, don't bother on the end tag.
            (cmarko::Event::End(_), _) | (_, cmarkn::Event::End(_)) if reported_an_error => {}
            // Non-matching inline.
            (cmarko::Event::Start(cmarko::Tag::Link(..)), cmarkn::Event::FootnoteReference(..))
            | (
                cmarko::Event::Start(cmarko::Tag::Image(..)),
                cmarkn::Event::FootnoteReference(..),
            )
            | (
                cmarko::Event::FootnoteReference(..),
                cmarkn::Event::Start(cmarkn::Tag::Link { .. }),
            )
            | (
                cmarko::Event::FootnoteReference(..),
                cmarkn::Event::Start(cmarkn::Tag::Image { .. }),
            ) if event_range_old == event_range_new => {
                reported_an_error = true;
                // If we can't get a span of the backtick, because it is in a `#[doc = ""]` attribute,
                // use the span of the entire attribute as a fallback.
                let span = source_span_for_markdown_range(
                    tcx,
                    &dox,
                    &event_range_old,
                    &item.attrs.doc_strings,
                )
                .unwrap_or_else(|| item.attr_span(tcx));
                tcx.node_span_lint(
                    crate::lint::UNPORTABLE_MARKDOWN,
                    hir_id,
                    span,
                    "unportable markdown",
                    |lint| {
                        lint.help(format!("syntax ambiguous between footnote and link"));
                    },
                );
            }
            // Non-matching results.
            (event_old, event_new) => {
                reported_an_error = true;
                let (range, range_other, desc, desc_other, tag, tag_other) = if event_range_old.end
                    - event_range_old.start
                    < event_range_new.end - event_range_new.start
                {
                    (
                        event_range_old,
                        event_range_new,
                        "old",
                        "new",
                        format!("{event_old:?}"),
                        format!("{event_new:?}"),
                    )
                } else {
                    (
                        event_range_new,
                        event_range_old,
                        "new",
                        "old",
                        format!("{event_new:?}"),
                        format!("{event_old:?}"),
                    )
                };
                let (range, tag_other) =
                    if range_other.start <= range.start && range_other.end <= range.end {
                        (range_other.start..range.end, tag_other)
                    } else {
                        (range, format!("nothing"))
                    };
                // If we can't get a span of the backtick, because it is in a `#[doc = ""]` attribute,
                // use the span of the entire attribute as a fallback.
                let span =
                    source_span_for_markdown_range(tcx, &dox, &range, &item.attrs.doc_strings)
                        .unwrap_or_else(|| item.attr_span(tcx));
                tcx.node_span_lint(
                    crate::lint::UNPORTABLE_MARKDOWN,
                    hir_id,
                    span,
                    "unportable markdown",
                    |lint| {
                        lint.help(format!(
                            "{desc} parser sees {tag}, {desc_other} sees {tag_other}"
                        ));
                    },
                );
            }
        }
    }
}
