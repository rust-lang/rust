use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::sugg::Sugg;
use clippy_utils::{ExprUseNode, expr_use_ctxt, std_or_core};
use rustc_errors::Applicability;
use rustc_hir::{Expr, Mutability, Ty, TyKind};
use rustc_lint::LateContext;
use rustc_middle::ty;

use super::REF_AS_PTR;

pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'_>,
    cast_expr: &'tcx Expr<'_>,
    cast_to_hir_ty: &Ty<'_>,
) {
    let (cast_from, cast_to) = (
        cx.typeck_results().expr_ty(cast_expr),
        cx.typeck_results().expr_ty(expr),
    );

    if matches!(cast_from.kind(), ty::Ref(..))
        && let ty::RawPtr(_, to_mutbl) = cast_to.kind()
        && let use_cx = expr_use_ctxt(cx, expr)
        // TODO: only block the lint if `cast_expr` is a temporary
        && !matches!(use_cx.use_node(cx), ExprUseNode::LetStmt(_) | ExprUseNode::ConstStatic(_))
        && let Some(std_or_core) = std_or_core(cx)
    {
        let fn_name = match to_mutbl {
            Mutability::Not => "from_ref",
            Mutability::Mut => "from_mut",
        };

        let mut app = Applicability::MachineApplicable;
        let turbofish = match &cast_to_hir_ty.kind {
            TyKind::Infer(()) => String::new(),
            TyKind::Ptr(mut_ty) => {
                if matches!(mut_ty.ty.kind, TyKind::Infer(())) {
                    String::new()
                } else {
                    format!(
                        "::<{}>",
                        snippet_with_applicability(cx, mut_ty.ty.span, "/* type */", &mut app)
                    )
                }
            },
            _ => return,
        };

        let cast_expr_sugg = Sugg::hir_with_applicability(cx, cast_expr, "_", &mut app);

        span_lint_and_sugg(
            cx,
            REF_AS_PTR,
            expr.span,
            "reference as raw pointer",
            "try",
            format!("{std_or_core}::ptr::{fn_name}{turbofish}({cast_expr_sugg})"),
            app,
        );
    }
}
