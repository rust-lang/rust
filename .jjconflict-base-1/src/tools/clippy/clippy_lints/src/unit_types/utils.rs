use rustc_hir::{Expr, ExprKind};

pub(super) fn is_unit_literal(expr: &Expr<'_>) -> bool {
    matches!(expr.kind, ExprKind::Tup(slice) if slice.is_empty())
}
