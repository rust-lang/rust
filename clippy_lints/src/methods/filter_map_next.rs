use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::res::{MaybeDef as _, MaybeTypeckRes as _};
use clippy_utils::source::snippet_with_applicability;
use rustc_errors::Applicability;
use rustc_hir::Expr;
use rustc_lint::LateContext;
use rustc_span::sym;

use super::FILTER_MAP_NEXT;

pub(super) fn check(cx: &LateContext<'_>, expr: &Expr<'_>, recv: &Expr<'_>, arg: &Expr<'_>, msrv: Msrv) {
    if cx.ty_based_def(expr).opt_parent(cx).is_diag_item(cx, sym::Iterator) && msrv.meets(cx, msrvs::ITERATOR_FIND_MAP)
    {
        span_lint_and_then(
            cx,
            FILTER_MAP_NEXT,
            expr.span,
            "called `filter_map(..).next()` on an `Iterator`",
            |diag| {
                let mut app = Applicability::MachineApplicable;
                let iter_snippet = snippet_with_applicability(cx, recv.span, "_", &mut app);
                let filter_snippet = snippet_with_applicability(cx, arg.span, "..", &mut app);
                diag.span_suggestion_verbose(
                    expr.span,
                    "use `.find_map(..)` instead",
                    format!("{iter_snippet}.find_map({filter_snippet})"),
                    app,
                );
            },
        );
    }
}
