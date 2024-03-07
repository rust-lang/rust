use super::DEPRECATED_SEMVER;
use clippy_utils::diagnostics::span_lint;
use rustc_ast::{LitKind, MetaItemLit};
use rustc_lint::LateContext;
use rustc_span::Span;
use semver::Version;

pub(super) fn check(cx: &LateContext<'_>, span: Span, lit: &MetaItemLit) {
    if let LitKind::Str(is, _) = lit.kind {
        if is.as_str() == "TBD" || Version::parse(is.as_str()).is_ok() {
            return;
        }
    }
    span_lint(
        cx,
        DEPRECATED_SEMVER,
        span,
        "the since field must contain a semver-compliant version",
    );
}
