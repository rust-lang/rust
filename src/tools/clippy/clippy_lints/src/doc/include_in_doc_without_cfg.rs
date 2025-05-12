use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet_opt;
use rustc_ast::{AttrArgs, AttrKind, AttrStyle, Attribute};
use rustc_errors::Applicability;
use rustc_lint::EarlyContext;

use super::DOC_INCLUDE_WITHOUT_CFG;

pub fn check(cx: &EarlyContext<'_>, attrs: &[Attribute]) {
    for attr in attrs {
        if !attr.span.from_expansion()
            && let AttrKind::Normal(ref item) = attr.kind
            && attr.doc_str().is_some()
            && let AttrArgs::Eq { expr: meta, .. } = &item.item.args
            && !attr.span.contains(meta.span)
            // Since the `include_str` is already expanded at this point, we can only take the
            // whole attribute snippet and then modify for our suggestion.
            && let Some(snippet) = snippet_opt(cx, attr.span)
            // We cannot remove this because a `#[doc = include_str!("...")]` attribute can occupy
            // several lines.
            && let Some(start) = snippet.find('[')
            && let Some(end) = snippet.rfind(']')
            && let snippet = &snippet[start + 1..end]
            // We check that the expansion actually comes from `include_str!` and not just from
            // another macro.
            && let Some(sub_snippet) = snippet.trim().strip_prefix("doc")
            && let Some(sub_snippet) = sub_snippet.trim().strip_prefix("=")
            && sub_snippet.trim().starts_with("include_str!")
        {
            span_lint_and_sugg(
                cx,
                DOC_INCLUDE_WITHOUT_CFG,
                attr.span,
                "included a file in documentation unconditionally",
                "use `cfg_attr(doc, doc = \"...\")`",
                format!(
                    "#{}[cfg_attr(doc, {snippet})]",
                    if attr.style == AttrStyle::Inner { "!" } else { "" }
                ),
                Applicability::MachineApplicable,
            );
        }
    }
}
