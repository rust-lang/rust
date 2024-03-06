use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::sugg::Sugg;
use clippy_utils::{expr_use_ctxt, is_no_std_crate, ExprUseNode};
use rustc_errors::Applicability;
use rustc_hir::{Expr, Mutability, Ty, TyKind};
use rustc_lint::LateContext;
use rustc_middle::ty::{self, TypeAndMut};

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
        && let ty::RawPtr(TypeAndMut { mutbl: to_mutbl, .. }) = cast_to.kind()
        && let Some(use_cx) = expr_use_ctxt(cx, expr)
        // TODO: only block the lint if `cast_expr` is a temporary
        && !matches!(use_cx.node, ExprUseNode::Local(_) | ExprUseNode::ConstStatic(_))
    {
        let core_or_std = if is_no_std_crate(cx) { "core" } else { "std" };
        let fn_name = match to_mutbl {
            Mutability::Not => "from_ref",
            Mutability::Mut => "from_mut",
        };

        let mut app = Applicability::MachineApplicable;
        let turbofish = match &cast_to_hir_ty.kind {
            TyKind::Infer => String::new(),
            TyKind::Ptr(mut_ty) => {
                if matches!(mut_ty.ty.kind, TyKind::Infer) {
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
            format!("{core_or_std}::ptr::{fn_name}{turbofish}({cast_expr_sugg})"),
            app,
        );
    }
}
