use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::source::SpanRangeExt;
use clippy_utils::ty::implements_trait;
use clippy_utils::{is_from_proc_macro, is_trait_method};
use rustc_errors::Applicability;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::{Expr, ExprKind};
use rustc_lint::{LateContext, LintContext};
use rustc_span::{Span, sym};

use super::MANUAL_TRY_FOLD;

pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &Expr<'tcx>,
    init: &Expr<'_>,
    acc: &Expr<'_>,
    fold_span: Span,
    msrv: Msrv,
) {
    if !fold_span.in_external_macro(cx.sess().source_map())
        && is_trait_method(cx, expr, sym::Iterator)
        && let init_ty = cx.typeck_results().expr_ty(init)
        && let Some(try_trait) = cx.tcx.lang_items().try_trait()
        && implements_trait(cx, init_ty, try_trait, &[])
        && let ExprKind::Call(path, [first, rest @ ..]) = init.kind
        && let ExprKind::Path(qpath) = path.kind
        && let Res::Def(DefKind::Ctor(_, _), _) = cx.qpath_res(&qpath, path.hir_id)
        && let ExprKind::Closure(closure) = acc.kind
        && msrv.meets(cx, msrvs::ITERATOR_TRY_FOLD)
        && !is_from_proc_macro(cx, expr)
        && let Some(args_snip) = closure
            .fn_arg_span
            .and_then(|fn_arg_span| fn_arg_span.get_source_text(cx))
    {
        let init_snip = rest
            .is_empty()
            .then_some(first.span)
            .and_then(|span| span.get_source_text(cx))
            .map_or_else(|| "...".to_owned(), |src| src.to_owned());

        span_lint_and_sugg(
            cx,
            MANUAL_TRY_FOLD,
            fold_span,
            "usage of `Iterator::fold` on a type that implements `Try`",
            "use `try_fold` instead",
            format!("try_fold({init_snip}, {args_snip} ...)",),
            Applicability::HasPlaceholders,
        );
    }
}
