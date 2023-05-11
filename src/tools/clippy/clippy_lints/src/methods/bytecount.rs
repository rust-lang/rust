use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::ty::match_type;
use clippy_utils::visitors::is_local_used;
use clippy_utils::{path_to_local_id, paths, peel_blocks, peel_ref_operators, strip_pat_refs};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::{BinOpKind, Closure, Expr, ExprKind, PatKind};
use rustc_lint::LateContext;
use rustc_middle::ty::{self, UintTy};
use rustc_span::sym;

use super::NAIVE_BYTECOUNT;

pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'_>,
    filter_recv: &'tcx Expr<'_>,
    filter_arg: &'tcx Expr<'_>,
) {
    if_chain! {
        if let ExprKind::Closure(&Closure { body, .. }) = filter_arg.kind;
        let body = cx.tcx.hir().body(body);
        if let [param] = body.params;
        if let PatKind::Binding(_, arg_id, _, _) = strip_pat_refs(param.pat).kind;
        if let ExprKind::Binary(ref op, l, r) = body.value.kind;
        if op.node == BinOpKind::Eq;
        if match_type(cx,
                    cx.typeck_results().expr_ty(filter_recv).peel_refs(),
                    &paths::SLICE_ITER);
        let operand_is_arg = |expr| {
            let expr = peel_ref_operators(cx, peel_blocks(expr));
            path_to_local_id(expr, arg_id)
        };
        let needle = if operand_is_arg(l) {
            r
        } else if operand_is_arg(r) {
            l
        } else {
            return;
        };
        if ty::Uint(UintTy::U8) == *cx.typeck_results().expr_ty(needle).peel_refs().kind();
        if !is_local_used(cx, needle, arg_id);
        then {
            let haystack = if let ExprKind::MethodCall(path, receiver, [], _) =
                    filter_recv.kind {
                let p = path.ident.name;
                if p == sym::iter || p == sym!(iter_mut) {
                    receiver
                } else {
                    filter_recv
                }
            } else {
                filter_recv
            };
            let mut applicability = Applicability::MaybeIncorrect;
            span_lint_and_sugg(
                cx,
                NAIVE_BYTECOUNT,
                expr.span,
                "you appear to be counting bytes the naive way",
                "consider using the bytecount crate",
                format!("bytecount::count({}, {})",
                        snippet_with_applicability(cx, haystack.span, "..", &mut applicability),
                        snippet_with_applicability(cx, needle.span, "..", &mut applicability)),
                applicability,
            );
        }
    };
}
