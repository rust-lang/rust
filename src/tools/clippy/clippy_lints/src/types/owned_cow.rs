use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet_opt;
use rustc_errors::Applicability;
use rustc_hir::def_id::DefId;
use rustc_hir::{self as hir};
use rustc_lint::LateContext;
use rustc_span::{Span, sym};

pub(super) fn check(cx: &LateContext<'_>, qpath: &hir::QPath<'_>, def_id: DefId) -> bool {
    if cx.tcx.is_diagnostic_item(sym::Cow, def_id)
        && let hir::QPath::Resolved(_, path) = qpath
        && let [.., last_seg] = path.segments
        && let Some(args) = last_seg.args
        && let [_lt, carg] = args.args
        && let hir::GenericArg::Type(cty) = carg
        && let Some((span, repl)) = replacement(cx, cty.as_unambig_ty())
    {
        span_lint_and_sugg(
            cx,
            super::OWNED_COW,
            span,
            "needlessly owned Cow type",
            "use",
            repl,
            Applicability::Unspecified,
        );
        return true;
    }
    false
}

fn replacement(cx: &LateContext<'_>, cty: &hir::Ty<'_>) -> Option<(Span, String)> {
    if clippy_utils::is_path_lang_item(cx, cty, hir::LangItem::String) {
        return Some((cty.span, "str".into()));
    }
    if clippy_utils::is_path_diagnostic_item(cx, cty, sym::Vec) {
        return if let hir::TyKind::Path(hir::QPath::Resolved(_, path)) = cty.kind
            && let [.., last_seg] = path.segments
            && let Some(args) = last_seg.args
            && let [t, ..] = args.args
            && let Some(snip) = snippet_opt(cx, t.span())
        {
            Some((cty.span, format!("[{snip}]")))
        } else {
            None
        };
    }
    if clippy_utils::is_path_diagnostic_item(cx, cty, sym::cstring_type) {
        return Some((
            cty.span,
            (if clippy_utils::is_no_std_crate(cx) {
                "core::ffi::CStr"
            } else {
                "std::ffi::CStr"
            })
            .into(),
        ));
    }
    // Neither OsString nor PathBuf are available outside std
    for (diag, repl) in [(sym::OsString, "std::ffi::OsStr"), (sym::PathBuf, "std::path::Path")] {
        if clippy_utils::is_path_diagnostic_item(cx, cty, diag) {
            return Some((cty.span, repl.into()));
        }
    }
    None
}
