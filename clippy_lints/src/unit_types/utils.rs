use rustc_hir::{Expr, ExprKind};
use rustc_middle::ty::{self, Ty};

pub(super) fn is_unit(ty: Ty<'_>) -> bool {
    matches!(ty.kind(), ty::Tuple(slice) if slice.is_empty())
}

pub(super) fn is_unit_literal(expr: &Expr<'_>) -> bool {
    matches!(expr.kind, ExprKind::Tup(ref slice) if slice.is_empty())
}
