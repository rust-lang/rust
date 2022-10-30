use super::MISSING_SPIN_LOOP;
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::is_no_std_crate;
use rustc_errors::Applicability;
use rustc_hir::{Block, Expr, ExprKind};
use rustc_lint::LateContext;
use rustc_middle::ty;
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
    if_chain! {
        if let ExprKind::Block(Block { stmts: [], expr: None, ..}, _) = body.kind;
        if let ExprKind::MethodCall(method, callee, ..) = unpack_cond(cond).kind;
        if [sym::load, sym::compare_exchange, sym::compare_exchange_weak].contains(&method.ident.name);
        if let ty::Adt(def, _substs) = cx.typeck_results().expr_ty(callee).kind();
        if cx.tcx.is_diagnostic_item(sym::AtomicBool, def.did());
        then {
            span_lint_and_sugg(
                cx,
                MISSING_SPIN_LOOP,
                body.span,
                "busy-waiting loop should at least have a spin loop hint",
                "try this",
                (if is_no_std_crate(cx) {
                    "{ core::hint::spin_loop() }"
                } else {
                    "{ std::hint::spin_loop() }"
                }).into(),
                Applicability::MachineApplicable
            );
        }
    }
}
