use clippy_utils::diagnostics::span_lint;
use rustc_ast::{FormatArgs, FormatArgsPiece, FormatPlaceholder, FormatTrait};
use rustc_lint::LateContext;

use super::USE_DEBUG;

pub(super) fn check(cx: &LateContext<'_>, format_args: &FormatArgs) {
    for piece in &format_args.template {
        if let &FormatArgsPiece::Placeholder(FormatPlaceholder {
            span: Some(span),
            format_trait: FormatTrait::Debug,
            ..
        }) = piece
        {
            span_lint(cx, USE_DEBUG, span, "use of `Debug`-based formatting");
        }
    }
}
