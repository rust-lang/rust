use crate::methods::STRING_FROM_UTF8_AS_BYTES;
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::res::{MaybeDef as _, MaybeQPath as _};
use clippy_utils::source::{snippet_with_applicability, snippet_with_context};
use clippy_utils::{method_calls, sym};
use rustc_ast::BorrowKind;
use rustc_errors::Applicability;
use rustc_hir::attrs::lang_items::LangItem;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::LateContext;

pub(super) fn check_call(cx: &LateContext<'_>, expr: &Expr<'_>, fun: &Expr<'_>, args: &[Expr<'_>]) {
    // Find `std::str::converts::from_utf8` or `std::primitive::str::from_utf8`

    if let [bytes_arg] = args && let Some(sym::str_from_utf8 | sym::str_inherent_from_utf8) =
        fun.res(cx).opt_diag_name(cx)

        // Find string::as_bytes
        && let ExprKind::AddrOf(BorrowKind::Ref, _, inner) = bytes_arg.kind
        && let ExprKind::Index(left, right, _) = inner.kind
        && let (method_names, expressions, _) = method_calls(left, 1)
        && method_names == [sym::as_bytes]
        && expressions.len() == 1
        && expressions[0].1.is_empty()

        // Check for slicer
        && let ExprKind::Struct(&qpath, _, _) = right.kind
        && cx.tcx.qpath_is_lang_item(qpath, LangItem::Range)
    {
        let mut applicability = Applicability::MachineApplicable;
        let string_expression = &expressions[0].0;

        let snippet_app = snippet_with_applicability(cx, string_expression.span, "..", &mut applicability);
        let (right_snip, _) = snippet_with_context(cx, right.span, expr.span.ctxt(), "..", &mut applicability);

        span_lint_and_sugg(
            cx,
            STRING_FROM_UTF8_AS_BYTES,
            expr.span,
            "calling a slice of `as_bytes()` with `from_utf8` should be not necessary",
            "try",
            format!("Some(&{snippet_app}[{right_snip}])"),
            applicability,
        );
    }
}
