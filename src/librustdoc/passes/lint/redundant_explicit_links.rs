use std::ops::Range;

use pulldown_cmark::{BrokenLink, CowStr, Event, LinkType, OffsetIter, Parser, Tag};
use rustc_ast::NodeId;
use rustc_errors::SuggestionStyle;
use rustc_hir::def::{DefKind, DocLinkResMap, Namespace, Res};
use rustc_hir::HirId;
use rustc_lint_defs::Applicability;
use rustc_resolve::rustdoc::source_span_for_markdown_range;
use rustc_span::Symbol;

use crate::clean::utils::find_nearest_parent_module;
use crate::clean::utils::inherits_doc_hidden;
use crate::clean::Item;
use crate::core::DocContext;
use crate::html::markdown::main_body_opts;

#[derive(Debug)]
struct LinkData {
    resolvable_link: Option<String>,
    resolvable_link_range: Option<Range<usize>>,
    display_link: String,
}

pub(crate) fn visit_item(cx: &DocContext<'_>, item: &Item) {
    let Some(hir_id) = DocContext::as_local_hir_id(cx.tcx, item.item_id) else {
        // If non-local, no need to check anything.
        return;
    };

    let doc = item.doc_value();
    if doc.is_empty() {
        return;
    }

    if item.link_names(&cx.cache).is_empty() {
        // If there's no link names in this item,
        // then we skip resolution querying to
        // avoid from panicking.
        return;
    }

    let Some(item_id) = item.def_id() else {
        return;
    };
    let Some(local_item_id) = item_id.as_local() else {
        return;
    };

    let is_hidden = !cx.render_options.document_hidden
        && (item.is_doc_hidden() || inherits_doc_hidden(cx.tcx, local_item_id, None));
    if is_hidden {
        return;
    }
    let is_private = !cx.render_options.document_private
        && !cx.cache.effective_visibilities.is_directly_public(cx.tcx, item_id);
    if is_private {
        return;
    }

    check_redundant_explicit_link(cx, item, hir_id, &doc);
}

fn check_redundant_explicit_link<'md>(
    cx: &DocContext<'_>,
    item: &Item,
    hir_id: HirId,
    doc: &'md str,
) -> Option<()> {
    let mut broken_line_callback = |link: BrokenLink<'md>| Some((link.reference, "".into()));
    let mut offset_iter = Parser::new_with_broken_link_callback(
        &doc,
        main_body_opts(),
        Some(&mut broken_line_callback),
    )
    .into_offset_iter();
    let item_id = item.def_id()?;
    let module_id = match cx.tcx.def_kind(item_id) {
        DefKind::Mod if item.inner_docs(cx.tcx) => item_id,
        _ => find_nearest_parent_module(cx.tcx, item_id).unwrap(),
    };
    let resolutions = cx.tcx.doc_link_resolutions(module_id);

    while let Some((event, link_range)) = offset_iter.next() {
        match event {
            Event::Start(Tag::Link(link_type, dest, _)) => {
                let link_data = collect_link_data(&mut offset_iter);

                if let Some(resolvable_link) = link_data.resolvable_link.as_ref() {
                    if &link_data.display_link.replace("`", "") != resolvable_link {
                        // Skips if display link does not match to actual
                        // resolvable link, usually happens if display link
                        // has several segments, e.g.
                        // [this is just an `Option`](Option)
                        continue;
                    }
                }

                let explicit_link = dest.to_string();
                let display_link = link_data.resolvable_link.clone()?;

                if explicit_link.ends_with(&display_link) || display_link.ends_with(&explicit_link)
                {
                    match link_type {
                        LinkType::Inline | LinkType::ReferenceUnknown => {
                            check_inline_or_reference_unknown_redundancy(
                                cx,
                                item,
                                hir_id,
                                doc,
                                resolutions,
                                link_range,
                                dest.to_string(),
                                link_data,
                                if link_type == LinkType::Inline {
                                    (b'(', b')')
                                } else {
                                    (b'[', b']')
                                },
                            );
                        }
                        LinkType::Reference => {
                            check_reference_redundancy(
                                cx,
                                item,
                                hir_id,
                                doc,
                                resolutions,
                                link_range,
                                &dest,
                                link_data,
                            );
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }

    None
}

/// FIXME(ChAoSUnItY): Too many arguments.
fn check_inline_or_reference_unknown_redundancy(
    cx: &DocContext<'_>,
    item: &Item,
    hir_id: HirId,
    doc: &str,
    resolutions: &DocLinkResMap,
    link_range: Range<usize>,
    dest: String,
    link_data: LinkData,
    (open, close): (u8, u8),
) -> Option<()> {
    let (resolvable_link, resolvable_link_range) =
        (&link_data.resolvable_link?, &link_data.resolvable_link_range?);
    let (dest_res, display_res) =
        (find_resolution(resolutions, &dest)?, find_resolution(resolutions, resolvable_link)?);

    if dest_res == display_res {
        let link_span =
            source_span_for_markdown_range(cx.tcx, &doc, &link_range, &item.attrs.doc_strings)
                .unwrap_or(item.attr_span(cx.tcx));
        let explicit_span = source_span_for_markdown_range(
            cx.tcx,
            &doc,
            &offset_explicit_range(doc, link_range, open, close),
            &item.attrs.doc_strings,
        )?;
        let display_span = source_span_for_markdown_range(
            cx.tcx,
            &doc,
            &resolvable_link_range,
            &item.attrs.doc_strings,
        )?;

        cx.tcx.struct_span_lint_hir(crate::lint::REDUNDANT_EXPLICIT_LINKS, hir_id, explicit_span, "redundant explicit link target", |lint| {
            lint.span_label(explicit_span, "explicit target is redundant")
                .span_label(display_span, "because label contains path that resolves to same destination")
                .note("when a link's destination is not specified,\nthe label is used to resolve intra-doc links")
                .span_suggestion_with_style(link_span, "remove explicit link target", format!("[{}]", link_data.display_link), Applicability::MaybeIncorrect, SuggestionStyle::ShowAlways);

            lint
        });
    }

    None
}

/// FIXME(ChAoSUnItY): Too many arguments.
fn check_reference_redundancy(
    cx: &DocContext<'_>,
    item: &Item,
    hir_id: HirId,
    doc: &str,
    resolutions: &DocLinkResMap,
    link_range: Range<usize>,
    dest: &CowStr<'_>,
    link_data: LinkData,
) -> Option<()> {
    let (resolvable_link, resolvable_link_range) =
        (&link_data.resolvable_link?, &link_data.resolvable_link_range?);
    let (dest_res, display_res) =
        (find_resolution(resolutions, &dest)?, find_resolution(resolutions, resolvable_link)?);

    if dest_res == display_res {
        let link_span =
            source_span_for_markdown_range(cx.tcx, &doc, &link_range, &item.attrs.doc_strings)
                .unwrap_or(item.attr_span(cx.tcx));
        let explicit_span = source_span_for_markdown_range(
            cx.tcx,
            &doc,
            &offset_explicit_range(doc, link_range.clone(), b'[', b']'),
            &item.attrs.doc_strings,
        )?;
        let display_span = source_span_for_markdown_range(
            cx.tcx,
            &doc,
            &resolvable_link_range,
            &item.attrs.doc_strings,
        )?;
        let def_span = source_span_for_markdown_range(
            cx.tcx,
            &doc,
            &offset_reference_def_range(doc, dest, link_range),
            &item.attrs.doc_strings,
        )?;

        cx.tcx.struct_span_lint_hir(crate::lint::REDUNDANT_EXPLICIT_LINKS, hir_id, explicit_span, "redundant explicit link target", |lint| {
            lint.span_label(explicit_span, "explicit target is redundant")
                .span_label(display_span, "because label contains path that resolves to same destination")
                .span_note(def_span, "referenced explicit link target defined here")
                .note("when a link's destination is not specified,\nthe label is used to resolve intra-doc links")
                .span_suggestion_with_style(link_span, "remove explicit link target", format!("[{}]", link_data.display_link), Applicability::MaybeIncorrect, SuggestionStyle::ShowAlways);

            lint
        });
    }

    None
}

fn find_resolution(resolutions: &DocLinkResMap, path: &str) -> Option<Res<NodeId>> {
    [Namespace::TypeNS, Namespace::ValueNS, Namespace::MacroNS]
        .into_iter()
        .find_map(|ns| resolutions.get(&(Symbol::intern(path), ns)).copied().flatten())
}

/// Collects all neccessary data of link.
fn collect_link_data(offset_iter: &mut OffsetIter<'_, '_>) -> LinkData {
    let mut resolvable_link = None;
    let mut resolvable_link_range = None;
    let mut display_link = String::new();

    while let Some((event, range)) = offset_iter.next() {
        match event {
            Event::Text(code) => {
                let code = code.to_string();
                display_link.push_str(&code);
                resolvable_link = Some(code);
                resolvable_link_range = Some(range);
            }
            Event::Code(code) => {
                let code = code.to_string();
                display_link.push('`');
                display_link.push_str(&code);
                display_link.push('`');
                resolvable_link = Some(code);
                resolvable_link_range = Some(range);
            }
            Event::End(_) => {
                break;
            }
            _ => {}
        }
    }

    LinkData { resolvable_link, resolvable_link_range, display_link }
}

fn offset_explicit_range(md: &str, link_range: Range<usize>, open: u8, close: u8) -> Range<usize> {
    let mut open_brace = !0;
    let mut close_brace = !0;
    for (i, b) in md.as_bytes()[link_range.clone()].iter().copied().enumerate().rev() {
        let i = i + link_range.start;
        if b == close {
            close_brace = i;
            break;
        }
    }

    if close_brace < link_range.start || close_brace >= link_range.end {
        return link_range;
    }

    let mut nesting = 1;

    for (i, b) in md.as_bytes()[link_range.start..close_brace].iter().copied().enumerate().rev() {
        let i = i + link_range.start;
        if b == close {
            nesting += 1;
        }
        if b == open {
            nesting -= 1;
        }
        if nesting == 0 {
            open_brace = i;
            break;
        }
    }

    assert!(open_brace != close_brace);

    if open_brace < link_range.start || open_brace >= link_range.end {
        return link_range;
    }
    // do not actually include braces in the span
    (open_brace + 1)..close_brace
}

fn offset_reference_def_range(
    md: &str,
    dest: &CowStr<'_>,
    link_range: Range<usize>,
) -> Range<usize> {
    // For diagnostics, we want to underline the link's definition but `span` will point at
    // where the link is used. This is a problem for reference-style links, where the definition
    // is separate from the usage.

    match dest {
        // `Borrowed` variant means the string (the link's destination) may come directly from
        // the markdown text and we can locate the original link destination.
        // NOTE: LinkReplacer also provides `Borrowed` but possibly from other sources,
        // so `locate()` can fall back to use `span`.
        CowStr::Borrowed(s) => {
            // FIXME: remove this function once pulldown_cmark can provide spans for link definitions.
            unsafe {
                let s_start = dest.as_ptr();
                let s_end = s_start.add(s.len());
                let md_start = md.as_ptr();
                let md_end = md_start.add(md.len());
                if md_start <= s_start && s_end <= md_end {
                    let start = s_start.offset_from(md_start) as usize;
                    let end = s_end.offset_from(md_start) as usize;
                    start..end
                } else {
                    link_range
                }
            }
        }

        // For anything else, we can only use the provided range.
        CowStr::Boxed(_) | CowStr::Inlined(_) => link_range,
    }
}
