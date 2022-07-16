use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::is_lang_ctor;
use clippy_utils::source::snippet;

use rustc_errors::Applicability;
use rustc_hir::LangItem::{OptionNone, OptionSome};
use rustc_hir::{Expr, ExprKind};
use rustc_lint::LateContext;

use super::{ITER_EMPTY, ITER_ONCE};

pub(super) fn check<'tcx>(cx: &LateContext<'tcx>, expr: &Expr<'_>, method_name: &str, recv: &Expr<'_>) {
    let item = match &recv.kind {
        ExprKind::Array(v) if v.len() <= 1 => v.first(),
        ExprKind::Path(p) => {
            if is_lang_ctor(cx, p, OptionNone) {
                None
            } else {
                return;
            }
        },
        ExprKind::Call(f, some_args) if some_args.len() == 1 => {
            if let ExprKind::Path(p) = &f.kind {
                if is_lang_ctor(cx, p, OptionSome) {
                    Some(&some_args[0])
                } else {
                    return;
                }
            } else {
                return;
            }
        },
        _ => return,
    };

    if let Some(i) = item {
        let (sugg, msg) = match method_name {
            "iter" => (
                format!("std::iter::once(&{})", snippet(cx, i.span, "...")),
                "this `iter` call can be replaced with std::iter::once",
            ),
            "iter_mut" => (
                format!("std::iter::once(&mut {})", snippet(cx, i.span, "...")),
                "this `iter_mut` call can be replaced with std::iter::once",
            ),
            "into_iter" => (
                format!("std::iter::once({})", snippet(cx, i.span, "...")),
                "this `into_iter` call can be replaced with std::iter::once",
            ),
            _ => return,
        };
        span_lint_and_sugg(cx, ITER_ONCE, expr.span, msg, "try", sugg, Applicability::Unspecified);
    } else {
        let msg = match method_name {
            "iter" => "this `iter call` can be replaced with std::iter::empty",
            "iter_mut" => "this `iter_mut` call can be replaced with std::iter::empty",
            "into_iter" => "this `into_iter` call can be replaced with std::iter::empty",
            _ => return,
        };
        span_lint_and_sugg(
            cx,
            ITER_EMPTY,
            expr.span,
            msg,
            "try",
            "std::iter::empty()".to_string(),
            Applicability::Unspecified,
        );
    }
}
