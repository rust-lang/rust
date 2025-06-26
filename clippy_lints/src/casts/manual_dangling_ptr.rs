use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::SpanRangeExt;
use clippy_utils::{expr_or_init, is_path_diagnostic_item, std_or_core, sym};
use rustc_ast::LitKind;
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind, GenericArg, Mutability, QPath, Ty, TyKind};
use rustc_lint::LateContext;
use rustc_span::source_map::Spanned;

use super::MANUAL_DANGLING_PTR;

pub(super) fn check(cx: &LateContext<'_>, expr: &Expr<'_>, from: &Expr<'_>, to: &Ty<'_>) {
    if let TyKind::Ptr(ref ptr_ty) = to.kind {
        let init_expr = expr_or_init(cx, from);
        if is_expr_const_aligned(cx, init_expr, ptr_ty.ty)
            && let Some(std_or_core) = std_or_core(cx)
        {
            let sugg_fn = match ptr_ty.mutbl {
                Mutability::Not => "ptr::dangling",
                Mutability::Mut => "ptr::dangling_mut",
            };

            let sugg = if let TyKind::Infer(()) = ptr_ty.ty.kind {
                format!("{std_or_core}::{sugg_fn}()")
            } else if let Some(mut_ty_snip) = ptr_ty.ty.span.get_source_text(cx) {
                format!("{std_or_core}::{sugg_fn}::<{mut_ty_snip}>()")
            } else {
                return;
            };

            span_lint_and_sugg(
                cx,
                MANUAL_DANGLING_PTR,
                expr.span,
                "manual creation of a dangling pointer",
                "use",
                sugg,
                Applicability::MachineApplicable,
            );
        }
    }
}

// Checks if the given expression is a call to `align_of` whose generic argument matches the target
// type, or a positive constant literal that matches the target type's alignment.
fn is_expr_const_aligned(cx: &LateContext<'_>, expr: &Expr<'_>, to: &Ty<'_>) -> bool {
    match expr.kind {
        ExprKind::Call(fun, _) => is_align_of_call(cx, fun, to),
        ExprKind::Lit(lit) => is_literal_aligned(cx, lit, to),
        _ => false,
    }
}

fn is_align_of_call(cx: &LateContext<'_>, fun: &Expr<'_>, to: &Ty<'_>) -> bool {
    if let ExprKind::Path(QPath::Resolved(_, path)) = fun.kind
        && is_path_diagnostic_item(cx, fun, sym::mem_align_of)
        && let Some(args) = path.segments.last().and_then(|seg| seg.args)
        && let [GenericArg::Type(generic_ty)] = args.args
    {
        let typeck = cx.typeck_results();
        return typeck.node_type(generic_ty.hir_id) == typeck.node_type(to.hir_id);
    }
    false
}

fn is_literal_aligned(cx: &LateContext<'_>, lit: &Spanned<LitKind>, to: &Ty<'_>) -> bool {
    let LitKind::Int(val, _) = lit.node else { return false };
    if val == 0 {
        return false;
    }
    let to_mid_ty = cx.typeck_results().node_type(to.hir_id);
    cx.tcx
        .layout_of(cx.typing_env().as_query_input(to_mid_ty))
        .is_ok_and(|layout| {
            let align = u128::from(layout.align.abi.bytes());
            u128::from(val) <= align
        })
}
