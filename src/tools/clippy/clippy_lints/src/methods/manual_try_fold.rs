use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::is_from_proc_macro;
use clippy_utils::msrvs::{Msrv, ITERATOR_TRY_FOLD};
use clippy_utils::source::snippet_opt;
use clippy_utils::ty::implements_trait;
use rustc_errors::Applicability;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::{Expr, ExprKind};
use rustc_lint::{LateContext, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_span::Span;

use super::MANUAL_TRY_FOLD;

pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &Expr<'tcx>,
    init: &Expr<'_>,
    acc: &Expr<'_>,
    fold_span: Span,
    msrv: &Msrv,
) {
    if !in_external_macro(cx.sess(), fold_span)
        && msrv.meets(ITERATOR_TRY_FOLD)
        && let init_ty = cx.typeck_results().expr_ty(init)
        && let Some(try_trait) = cx.tcx.lang_items().try_trait()
        && implements_trait(cx, init_ty, try_trait, &[])
        && let ExprKind::Call(path, [first, rest @ ..]) = init.kind
        && let ExprKind::Path(qpath) = path.kind
        && let Res::Def(DefKind::Ctor(_, _), _) = cx.qpath_res(&qpath, path.hir_id)
        && let ExprKind::Closure(closure) = acc.kind
        && !is_from_proc_macro(cx, expr)
        && let Some(args_snip) = closure.fn_arg_span.and_then(|fn_arg_span| snippet_opt(cx, fn_arg_span))
    {
        let init_snip = rest
            .is_empty()
            .then_some(first.span)
            .and_then(|span| snippet_opt(cx, span))
            .unwrap_or("...".to_owned());

        span_lint_and_sugg(
            cx,
            MANUAL_TRY_FOLD,
            fold_span,
            "usage of `Iterator::fold` on a type that implements `Try`",
            "use `try_fold` instead",
            format!("try_fold({init_snip}, {args_snip} ...)", ),
            Applicability::HasPlaceholders,
        );
    }
}
