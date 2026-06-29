use super::CHUNKS_EXACT_TO_AS_CHUNKS;
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::res::{MaybeDef, MaybeTypeckRes};
use clippy_utils::source::snippet_with_context;
use clippy_utils::visitors::is_const_evaluatable;
use clippy_utils::{get_expr_use_site, sym};
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind, Node, PatKind};
use rustc_lint::LateContext;
use rustc_middle::ty;
use rustc_span::{DesugaringKind, ExpnKind, Span, Symbol};

pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    recv: &'tcx Expr<'tcx>,
    arg: &'tcx Expr<'tcx>,
    expr: &'tcx Expr<'tcx>,
    call_span: Span,
    method_name: Symbol,
    msrv: Msrv,
) {
    let recv_ty = cx.typeck_results().expr_ty_adjusted(recv);
    if !matches!(recv_ty.kind(), ty::Ref(_, inner, _) if inner.is_slice()) {
        return;
    }

    if is_const_evaluatable(cx.tcx, cx.typeck_results(), arg) {
        if !msrv.meets(cx, msrvs::AS_CHUNKS) {
            return;
        }

        let use_ctxt = get_expr_use_site(cx.tcx, cx.typeck_results(), expr.span.ctxt(), expr);

        if use_ctxt.is_ty_unified {
            return;
        }

        let is_mut = method_name == sym::chunks_exact_mut;
        let suggestion_method = if is_mut { "as_chunks_mut" } else { "as_chunks" };
        let iter_method = if is_mut { "iter_mut" } else { "iter" };

        let mut applicability = Applicability::MachineApplicable;
        let arg_str = snippet_with_context(cx, arg.span, expr.span.ctxt(), "_", &mut applicability).0;

        let as_chunks = format_args!("{suggestion_method}::<{arg_str}>()");

        span_lint_and_then(
            cx,
            CHUNKS_EXACT_TO_AS_CHUNKS,
            call_span,
            format!("using `{method_name}` with a constant chunk size"),
            |diag| {
                if let Node::Expr(use_expr) = use_ctxt.node {
                    match use_expr.kind {
                        ExprKind::Call(_, [recv]) | ExprKind::MethodCall(_, recv, [], _)
                            if recv.hir_id == use_ctxt.child_id
                                && matches!(
                                    use_expr.span.ctxt().outer_expn_data().kind,
                                    ExpnKind::Desugaring(DesugaringKind::ForLoop),
                                ) =>
                        {
                            diag.span_suggestion(
                                call_span,
                                format!("consider using `{suggestion_method}` instead"),
                                format!("{as_chunks}.0"),
                                applicability,
                            );
                            return;
                        },
                        ExprKind::MethodCall(_, recv, ..)
                            if recv.hir_id == use_ctxt.child_id
                                && cx
                                    .ty_based_def(use_expr)
                                    .assoc_fn_parent(cx)
                                    .is_diag_item(cx, sym::Iterator) =>
                        {
                            diag.span_suggestion(
                                call_span,
                                format!("consider using `{suggestion_method}` instead"),
                                format!("{as_chunks}.0.{iter_method}()"),
                                applicability,
                            );
                            return;
                        },
                        _ => {},
                    }
                }

                diag.span_help(call_span, format!("consider using `{as_chunks}` instead"));

                if let Node::LetStmt(let_stmt) = use_ctxt.node
                    && let PatKind::Binding(_, _, ident, _) = let_stmt.pat.kind
                {
                    diag.note(format!(
                        "you can access the chunks using `{ident}.0.{iter_method}()`, and the remainder using `{ident}.1`"
                    ));
                }
            },
        );
    }
}
