use clippy_utils::diagnostics::span_lint;
use clippy_utils::res::MaybeDef;
use rustc_hir as hir;
use rustc_lint::LateContext;
use rustc_middle::ty;
use rustc_span::sym;

use super::ZST_OFFSET;

pub(super) fn check(cx: &LateContext<'_>, expr: &hir::Expr<'_>, recv: &hir::Expr<'_>) {
    let recv_ty = cx.typeck_results().expr_ty(recv);
    let pointee_ty = match recv_ty.kind() {
        ty::RawPtr(ty, _) => *ty,
        ty::Adt(_, args) if recv_ty.is_diag_item(cx, sym::NonNull) => args.type_at(0),
        _ => return,
    };
    if let Ok(layout) = cx.tcx.layout_of(cx.typing_env().as_query_input(pointee_ty))
        && layout.is_zst()
    {
        span_lint(cx, ZST_OFFSET, expr.span, "offset calculation on zero-sized value");
    }
}
