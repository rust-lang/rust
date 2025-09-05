use clippy_utils::diagnostics::span_lint;
use clippy_utils::ty::is_c_void;
use clippy_utils::{get_parent_expr, is_hir_ty_cfg_dependant, sym};
use rustc_hir::{Expr, ExprKind, GenericArg};
use rustc_lint::LateContext;
use rustc_middle::ty::layout::LayoutOf;
use rustc_middle::ty::{self, Ty};

use super::CAST_PTR_ALIGNMENT;

pub(super) fn check<'tcx>(cx: &LateContext<'tcx>, expr: &Expr<'_>, cast_from: Ty<'tcx>, cast_to: Ty<'tcx>) {
    lint_cast_ptr_alignment(cx, expr, cast_from, cast_to);
}

pub(super) fn check_cast_method(cx: &LateContext<'_>, expr: &Expr<'_>) {
    if let ExprKind::MethodCall(method_path, self_arg, [], _) = &expr.kind
        && method_path.ident.name == sym::cast
        && let Some(generic_args) = method_path.args
        && let [GenericArg::Type(cast_to)] = generic_args.args
        // There probably is no obvious reason to do this, just to be consistent with `as` cases.
        && !is_hir_ty_cfg_dependant(cx, cast_to.as_unambig_ty())
    {
        let (cast_from, cast_to) = (cx.typeck_results().expr_ty(self_arg), cx.typeck_results().expr_ty(expr));
        lint_cast_ptr_alignment(cx, expr, cast_from, cast_to);
    }
}

fn lint_cast_ptr_alignment<'tcx>(cx: &LateContext<'tcx>, expr: &Expr<'_>, cast_from: Ty<'tcx>, cast_to: Ty<'tcx>) {
    if let ty::RawPtr(from_ptr_ty, _) = *cast_from.kind()
        && let ty::RawPtr(to_ptr_ty, _) = *cast_to.kind()
        && let Ok(from_layout) = cx.layout_of(from_ptr_ty)
        && let Ok(to_layout) = cx.layout_of(to_ptr_ty)
        && from_layout.align.abi < to_layout.align.abi
        // with c_void, we inherently need to trust the user
        && !is_c_void(cx, from_ptr_ty)
        // when casting from a ZST, we don't know enough to properly lint
        && !from_layout.is_zst()
        && !is_used_as_unaligned(cx, expr)
    {
        span_lint(
            cx,
            CAST_PTR_ALIGNMENT,
            expr.span,
            format!(
                "casting from `{cast_from}` to a more-strictly-aligned pointer (`{cast_to}`) ({} < {} bytes)",
                from_layout.align.abi.bytes(),
                to_layout.align.abi.bytes(),
            ),
        );
    }
}

fn is_used_as_unaligned(cx: &LateContext<'_>, e: &Expr<'_>) -> bool {
    let Some(parent) = get_parent_expr(cx, e) else {
        return false;
    };
    match parent.kind {
        ExprKind::MethodCall(name, self_arg, ..) if self_arg.hir_id == e.hir_id => {
            if matches!(name.ident.name, sym::read_unaligned | sym::write_unaligned)
                && let Some(def_id) = cx.typeck_results().type_dependent_def_id(parent.hir_id)
                && let Some(def_id) = cx.tcx.impl_of_assoc(def_id)
                && cx.tcx.type_of(def_id).instantiate_identity().is_raw_ptr()
            {
                true
            } else {
                false
            }
        },
        ExprKind::Call(func, [arg, ..]) if arg.hir_id == e.hir_id => {
            if let ExprKind::Path(path) = &func.kind
                && let Some(def_id) = cx.qpath_res(path, func.hir_id).opt_def_id()
                && let Some(name) = cx.tcx.get_diagnostic_name(def_id)
                && matches!(
                    name,
                    sym::ptr_write_unaligned
                        | sym::ptr_read_unaligned
                        | sym::intrinsics_unaligned_volatile_load
                        | sym::intrinsics_unaligned_volatile_store
                )
            {
                true
            } else {
                false
            }
        },
        _ => false,
    }
}
