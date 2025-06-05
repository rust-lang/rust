use clippy_utils::diagnostics::span_lint_and_then;
use rustc_ast::token::CommentKind;
use rustc_errors::Applicability;
use rustc_hir::{AttrStyle, Attribute};
use rustc_lint::{LateContext, LintContext};
use rustc_resolve::rustdoc::DocFragmentKind;

use std::ops::Range;

use super::{DOC_SUSPICIOUS_FOOTNOTES, Fragments};

pub fn check(cx: &LateContext<'_>, doc: &str, range: Range<usize>, fragments: &Fragments<'_>, attrs: &[Attribute]) {
    for i in doc[range.clone()]
        .bytes()
        .enumerate()
        .filter_map(|(i, c)| if c == b'[' { Some(i) } else { None })
    {
        let start = i + range.start;
        if doc.as_bytes().get(start + 1) == Some(&b'^')
            && let Some(end) = all_numbers_upto_brace(doc, start + 2)
            && doc.as_bytes().get(end) != Some(&b':')
            && doc.as_bytes().get(start - 1) != Some(&b'\\')
            && let Some(this_fragment) = {
                // the `doc` string contains all fragments concatenated together
                // figure out which one this suspicious footnote comes from
                let mut starting_position = 0;
                let mut found_fragment = fragments.fragments.last();
                for fragment in fragments.fragments {
                    if start >= starting_position && start < starting_position + fragment.doc.as_str().len() {
                        found_fragment = Some(fragment);
                        break;
                    }
                    starting_position += fragment.doc.as_str().len();
                }
                found_fragment
            }
        {
            let span = fragments.span(cx, start..end).unwrap_or(this_fragment.span);
            span_lint_and_then(
                cx,
                DOC_SUSPICIOUS_FOOTNOTES,
                span,
                "looks like a footnote ref, but has no matching footnote",
                |diag| {
                    if this_fragment.kind == DocFragmentKind::SugaredDoc {
                        let (doc_attr, (_, doc_attr_comment_kind)) = attrs
                            .iter()
                            .filter(|attr| attr.span().overlaps(this_fragment.span))
                            .rev()
                            .find_map(|attr| Some((attr, attr.doc_str_and_comment_kind()?)))
                            .unwrap();
                        let (to_add, terminator) = match (doc_attr_comment_kind, doc_attr.style()) {
                            (CommentKind::Line, AttrStyle::Outer) => ("\n///\n/// ", ""),
                            (CommentKind::Line, AttrStyle::Inner) => ("\n//!\n//! ", ""),
                            (CommentKind::Block, AttrStyle::Outer) => ("\n/** ", " */"),
                            (CommentKind::Block, AttrStyle::Inner) => ("\n/*! ", " */"),
                        };
                        diag.span_suggestion_verbose(
                            doc_attr.span().shrink_to_hi(),
                            "add footnote definition",
                            format!(
                                "{to_add}{label}: <!-- description -->{terminator}",
                                label = &doc[start..end]
                            ),
                            Applicability::HasPlaceholders,
                        );
                    } else {
                        let is_file_include = cx
                            .sess()
                            .source_map()
                            .span_to_snippet(this_fragment.span)
                            .as_ref()
                            .map(|vdoc| vdoc.trim())
                            == Ok(doc);
                        if is_file_include {
                            // if this is a file include, then there's no quote marks
                            diag.span_suggestion_verbose(
                                this_fragment.span.shrink_to_hi(),
                                "add footnote definition",
                                format!("\n\n{label}: <!-- description -->", label = &doc[start..end],),
                                Applicability::HasPlaceholders,
                            );
                        } else {
                            // otherwise, we wrap in a string
                            diag.span_suggestion_verbose(
                                this_fragment.span,
                                "add footnote definition",
                                format!(
                                    "r#\"{doc}\n\n{label}: <!-- description -->\"#",
                                    doc = this_fragment.doc,
                                    label = &doc[start..end],
                                ),
                                Applicability::HasPlaceholders,
                            );
                        }
                    }
                },
            );
        }
    }
}

fn all_numbers_upto_brace(text: &str, i: usize) -> Option<usize> {
    for (j, c) in text.as_bytes()[i..].iter().copied().enumerate().take(64) {
        if c == b']' && j != 0 {
            return Some(i + j + 1);
        }
        if !c.is_ascii_digit() || j >= 64 {
            break;
        }
    }
    None
}
