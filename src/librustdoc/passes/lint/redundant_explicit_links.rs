use std::ops::Range;

use pulldown_cmark::{Parser, BrokenLink, Event, Tag, LinkType, OffsetIter};
use rustc_ast::NodeId;
use rustc_errors::SuggestionStyle;
use rustc_hir::HirId;
use rustc_hir::def::{Namespace, DefKind, DocLinkResMap, Res};
use rustc_lint_defs::Applicability;
use rustc_span::Symbol;

use crate::clean::Item;
use crate::clean::utils::find_nearest_parent_module;
use crate::core::DocContext;
use crate::html::markdown::main_body_opts;
use crate::passes::source_span_for_markdown_range;

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

    check_redundant_explicit_link(cx, item, hir_id, &doc);
}

fn check_redundant_explicit_link<'md>(cx: &DocContext<'_>, item: &Item, hir_id: HirId, doc: &'md str) {
    let mut broken_line_callback = |link: BrokenLink<'md>| Some((link.reference, "".into()));
    let mut offset_iter = Parser::new_with_broken_link_callback(&doc, main_body_opts(), Some(&mut broken_line_callback)).into_offset_iter();

    while let Some((event, link_range)) = offset_iter.next() {
        match event {
            Event::Start(Tag::Link(link_type, dest, _)) => {
                let link_data = collect_link_data(&mut offset_iter);
                let dest = dest.to_string();

                if link_type == LinkType::Inline {
                    check_inline_link_redundancy(cx, item, hir_id, doc, link_range, dest, link_data);
                }
            }
            _ => {}
        }
    }
}

fn check_inline_link_redundancy(cx: &DocContext<'_>, item: &Item, hir_id: HirId, doc: &str, link_range: Range<usize>, dest: String, link_data: LinkData) -> Option<()> {
    let item_id = item.def_id()?;
    let module_id = match cx.tcx.def_kind(item_id) {
        DefKind::Mod if item.inner_docs(cx.tcx) => item_id,
        _ => find_nearest_parent_module(cx.tcx, item_id).unwrap(),
    };
    let resolutions = cx.tcx.doc_link_resolutions(module_id);
    
    let (resolvable_link, resolvable_link_range) = (&link_data.resolvable_link?, &link_data.resolvable_link_range?);
    let (dest_res, display_res) = (find_resolution(resolutions, &dest)?, find_resolution(resolutions, resolvable_link)?);

    if dest_res == display_res {
        let link_span = source_span_for_markdown_range(
            cx.tcx,
            &doc,
            &link_range,
            &item.attrs,
        ).unwrap_or(item.attr_span(cx.tcx));
        let explicit_span = source_span_for_markdown_range(
            cx.tcx,
            &doc,
            &offset_explicit_range(doc, &link_range, b'(', b')'),
            &item.attrs
        )?;
        let display_span = source_span_for_markdown_range(
            cx.tcx,
            &doc,
            &resolvable_link_range,
            &item.attrs
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

fn find_resolution<'tcx>(resolutions: &'tcx DocLinkResMap, path: &str) -> Option<&'tcx Res<NodeId>> {
    for ns in [Namespace::TypeNS, Namespace::ValueNS, Namespace::MacroNS] {
        let Some(Some(res)) = resolutions.get(&(Symbol::intern(path), ns))
        else {
            continue;
        };

        return Some(res);
    }

    None
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

    LinkData {
        resolvable_link,
        resolvable_link_range,
        display_link,
    }
}

fn offset_explicit_range(md: &str, link_range: &Range<usize>, open: u8, close: u8) -> Range<usize> {
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
        return link_range.clone();
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
        return link_range.clone();
    }
    // do not actually include braces in the span
    (open_brace + 1)..close_brace
}
