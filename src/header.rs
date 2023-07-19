//! headers are sets of consecutive keywords and tokens, such as
//! `pub const unsafe fn foo` and `pub(crate) unsafe trait Bar`.
//!
//! This module contains general logic for formatting such headers,
//! where they are always placed on a single line except when there
//! are comments between parts of the header.

use std::borrow::Cow;

use rustc_ast as ast;
use rustc_span::symbol::Ident;
use rustc_span::Span;

use crate::comment::combine_strs_with_missing_comments;
use crate::rewrite::RewriteContext;
use crate::shape::Shape;
use crate::utils::rewrite_ident;

pub(crate) fn format_header(
    context: &RewriteContext<'_>,
    shape: Shape,
    parts: Vec<HeaderPart>,
) -> String {
    debug!(?parts, "format_header");
    let shape = shape.infinite_width();

    // Empty `HeaderPart`s are ignored.
    let mut parts = parts.into_iter().filter(|x| !x.snippet.is_empty());
    let Some(part) = parts.next() else {
        return String::new();
    };

    let mut result = part.snippet.into_owned();
    let mut span = part.span;

    for part in parts {
        debug!(?result, "before combine");
        result = combine_strs_with_missing_comments(
            context,
            &result,
            &part.snippet,
            span.between(part.span),
            shape,
            true,
        )
        .unwrap_or_else(|| format!("{} {}", &result, part.snippet));
        debug!(?result);
        span = part.span;
    }

    result
}

#[derive(Debug)]
pub(crate) struct HeaderPart {
    /// snippet of this part without surrounding space
    snippet: Cow<'static, str>,
    span: Span,
}

impl HeaderPart {
    pub(crate) fn new(snippet: impl Into<Cow<'static, str>>, span: Span) -> Self {
        Self {
            snippet: snippet.into(),
            span,
        }
    }

    pub(crate) fn ident(context: &RewriteContext<'_>, ident: Ident) -> Self {
        Self {
            snippet: rewrite_ident(context, ident).to_owned().into(),
            span: ident.span,
        }
    }

    pub(crate) fn visibility(context: &RewriteContext<'_>, vis: &ast::Visibility) -> Self {
        let snippet = match vis.kind {
            ast::VisibilityKind::Public => Cow::from("pub"),
            ast::VisibilityKind::Inherited => Cow::from(""),
            ast::VisibilityKind::Restricted { ref path, .. } => {
                let ast::Path { ref segments, .. } = **path;
                let mut segments_iter =
                    segments.iter().map(|seg| rewrite_ident(context, seg.ident));
                if path.is_global() {
                    segments_iter
                        .next()
                        .expect("Non-global path in pub(restricted)?");
                }
                let is_keyword = |s: &str| s == "crate" || s == "self" || s == "super";
                let path = segments_iter.collect::<Vec<_>>().join("::");
                let in_str = if is_keyword(&path) { "" } else { "in " };

                Cow::from(format!("pub({}{})", in_str, path))
            }
        };

        HeaderPart {
            snippet,
            span: vis.span,
        }
    }
}
