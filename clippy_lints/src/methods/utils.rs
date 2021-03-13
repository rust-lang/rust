use crate::utils::is_type_diagnostic_item;
use rustc_hir as hir;
use rustc_lint::LateContext;
use rustc_middle::ty;
use rustc_middle::ty::Ty;
use rustc_span::symbol::sym;

pub fn derefs_to_slice<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx hir::Expr<'tcx>,
    ty: Ty<'tcx>,
) -> Option<&'tcx hir::Expr<'tcx>> {
    fn may_slice<'a>(cx: &LateContext<'a>, ty: Ty<'a>) -> bool {
        match ty.kind() {
            ty::Slice(_) => true,
            ty::Adt(def, _) if def.is_box() => may_slice(cx, ty.boxed_ty()),
            ty::Adt(..) => is_type_diagnostic_item(cx, ty, sym::vec_type),
            ty::Array(_, size) => size
                .try_eval_usize(cx.tcx, cx.param_env)
                .map_or(false, |size| size < 32),
            ty::Ref(_, inner, _) => may_slice(cx, inner),
            _ => false,
        }
    }

    if let hir::ExprKind::MethodCall(ref path, _, ref args, _) = expr.kind {
        if path.ident.name == sym::iter && may_slice(cx, cx.typeck_results().expr_ty(&args[0])) {
            Some(&args[0])
        } else {
            None
        }
    } else {
        match ty.kind() {
            ty::Slice(_) => Some(expr),
            ty::Adt(def, _) if def.is_box() && may_slice(cx, ty.boxed_ty()) => Some(expr),
            ty::Ref(_, inner, _) => {
                if may_slice(cx, inner) {
                    Some(expr)
                } else {
                    None
                }
            },
            _ => None,
        }
    }
}
