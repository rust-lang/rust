use super::MISSING_SPIN_LOOP;
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::std_or_core;
use clippy_utils::ty::is_type_diagnostic_item;
use rustc_errors::Applicability;
use rustc_hir::{Block, Expr, ExprKind};
use rustc_lint::LateContext;
use rustc_span::sym;

fn unpack_cond<'tcx>(cond: &'tcx Expr<'tcx>) -> &'tcx Expr<'tcx> {
    match &cond.kind {
        ExprKind::Block(
            Block {
                stmts: [],
                expr: Some(e),
                ..
            },
            _,
        )
        | ExprKind::Unary(_, e) => unpack_cond(e),
        ExprKind::Binary(_, l, r) => {
            let l = unpack_cond(l);
            if let ExprKind::MethodCall(..) = l.kind {
                l
            } else {
                unpack_cond(r)
            }
        },
        _ => cond,
    }
}

pub(super) fn check<'tcx>(cx: &LateContext<'tcx>, cond: &'tcx Expr<'_>, body: &'tcx Expr<'_>) {
    if let ExprKind::Block(
        Block {
            stmts: [], expr: None, ..
        },
        _,
    ) = body.kind
        && let ExprKind::MethodCall(method, callee, ..) = unpack_cond(cond).kind
        && [sym::load, sym::compare_exchange, sym::compare_exchange_weak].contains(&method.ident.name)
        && let callee_ty = cx.typeck_results().expr_ty(callee)
        && is_type_diagnostic_item(cx, callee_ty, sym::AtomicBool)
        && let Some(std_or_core) = std_or_core(cx)
    {
        span_lint_and_sugg(
            cx,
            MISSING_SPIN_LOOP,
            body.span,
            "busy-waiting loop should at least have a spin loop hint",
            "try",
            format!("{{ {std_or_core}::hint::spin_loop() }}"),
            Applicability::MachineApplicable,
        );
    }
}
