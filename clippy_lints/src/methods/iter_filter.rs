use rustc_lint::LateContext;

use super::{ITER_FILTER_IS_OK, ITER_FILTER_IS_SOME};

use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::{indent_of, reindent_multiline};
use clippy_utils::{is_trait_method, peel_blocks};
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_hir::def::Res;
use rustc_hir::QPath;
use rustc_span::symbol::{sym, Symbol};
use rustc_span::Span;
use std::borrow::Cow;

fn is_method(cx: &LateContext<'_>, expr: &hir::Expr<'_>, method_name: Symbol) -> bool {
    match &expr.kind {
        hir::ExprKind::Path(QPath::TypeRelative(_, mname)) => mname.ident.name == method_name,
        hir::ExprKind::Path(QPath::Resolved(_, segments)) => {
            segments.segments.last().unwrap().ident.name == method_name
        },
        hir::ExprKind::MethodCall(segment, _, _, _) => segment.ident.name == method_name,
        hir::ExprKind::Closure(&hir::Closure { body, .. }) => {
            let body = cx.tcx.hir().body(body);
            let closure_expr = peel_blocks(body.value);
            let arg_id = body.params[0].pat.hir_id;
            match closure_expr.kind {
                hir::ExprKind::MethodCall(hir::PathSegment { ident, .. }, receiver, ..) => {
                    if ident.name == method_name
                        && let hir::ExprKind::Path(path) = &receiver.kind
                        && let Res::Local(ref local) = cx.qpath_res(path, receiver.hir_id)
                    {
                        return arg_id == *local;
                    }
                    false
                },
                _ => false,
            }
        },
        _ => false,
    }
}

fn parent_is_map(cx: &LateContext<'_>, expr: &hir::Expr<'_>) -> bool {
    if let hir::Node::Expr(parent_expr) = cx.tcx.hir().get_parent(expr.hir_id) {
        is_method(cx, parent_expr, rustc_span::sym::map)
    } else {
        false
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn check(cx: &LateContext<'_>, expr: &hir::Expr<'_>, filter_arg: &hir::Expr<'_>, filter_span: Span) {
    let is_iterator = is_trait_method(cx, expr, sym::Iterator);
    let parent_is_not_map = !parent_is_map(cx, expr);

    if is_iterator && parent_is_not_map && is_method(cx, filter_arg, sym!(is_some)) {
        span_lint_and_sugg(
            cx,
            ITER_FILTER_IS_SOME,
            filter_span.with_hi(expr.span.hi()),
            "`filter` for `is_some` on iterator over `Option`",
            "consider using `flatten` instead",
            reindent_multiline(Cow::Borrowed("flatten()"), true, indent_of(cx, filter_span)).into_owned(),
            Applicability::MaybeIncorrect,
        );
    }
    if is_iterator && parent_is_not_map && is_method(cx, filter_arg, sym!(is_ok)) {
        span_lint_and_sugg(
            cx,
            ITER_FILTER_IS_OK,
            filter_span.with_hi(expr.span.hi()),
            "`filter` for `is_ok` on iterator over `Result`s",
            "consider using `flatten` instead",
            reindent_multiline(Cow::Borrowed("flatten()"), true, indent_of(cx, filter_span)).into_owned(),
            Applicability::MaybeIncorrect,
        );
    }
}
