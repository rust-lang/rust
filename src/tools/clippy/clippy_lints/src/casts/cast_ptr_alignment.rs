use clippy_utils::diagnostics::span_lint;
use clippy_utils::is_hir_ty_cfg_dependant;
use if_chain::if_chain;
use rustc_hir::{Expr, ExprKind, GenericArg};
use rustc_lint::LateContext;
use rustc_middle::ty::layout::LayoutOf;
use rustc_middle::ty::{self, Ty};
use rustc_span::symbol::sym;

use super::CAST_PTR_ALIGNMENT;

pub(super) fn check(cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
    if let ExprKind::Cast(cast_expr, cast_to) = expr.kind {
        if is_hir_ty_cfg_dependant(cx, cast_to) {
            return;
        }
        let (cast_from, cast_to) = (
            cx.typeck_results().expr_ty(cast_expr),
            cx.typeck_results().expr_ty(expr),
        );
        lint_cast_ptr_alignment(cx, expr, cast_from, cast_to);
    } else if let ExprKind::MethodCall(method_path, _, args, _) = expr.kind {
        if_chain! {
            if method_path.ident.name == sym!(cast);
            if let Some(generic_args) = method_path.args;
            if let [GenericArg::Type(cast_to)] = generic_args.args;
            // There probably is no obvious reason to do this, just to be consistent with `as` cases.
            if !is_hir_ty_cfg_dependant(cx, cast_to);
            then {
                let (cast_from, cast_to) =
                    (cx.typeck_results().expr_ty(&args[0]), cx.typeck_results().expr_ty(expr));
                lint_cast_ptr_alignment(cx, expr, cast_from, cast_to);
            }
        }
    }
}

fn lint_cast_ptr_alignment<'tcx>(cx: &LateContext<'tcx>, expr: &Expr<'_>, cast_from: Ty<'tcx>, cast_to: Ty<'tcx>) {
    if_chain! {
        if let ty::RawPtr(from_ptr_ty) = &cast_from.kind();
        if let ty::RawPtr(to_ptr_ty) = &cast_to.kind();
        if let Ok(from_layout) = cx.layout_of(from_ptr_ty.ty);
        if let Ok(to_layout) = cx.layout_of(to_ptr_ty.ty);
        if from_layout.align.abi < to_layout.align.abi;
        // with c_void, we inherently need to trust the user
        if !is_c_void(cx, from_ptr_ty.ty);
        // when casting from a ZST, we don't know enough to properly lint
        if !from_layout.is_zst();
        then {
            span_lint(
                cx,
                CAST_PTR_ALIGNMENT,
                expr.span,
                &format!(
                    "casting from `{}` to a more-strictly-aligned pointer (`{}`) ({} < {} bytes)",
                    cast_from,
                    cast_to,
                    from_layout.align.abi.bytes(),
                    to_layout.align.abi.bytes(),
                ),
            );
        }
    }
}

/// Check if the given type is either `core::ffi::c_void` or
/// one of the platform specific `libc::<platform>::c_void` of libc.
fn is_c_void(cx: &LateContext<'_>, ty: Ty<'_>) -> bool {
    if let ty::Adt(adt, _) = ty.kind() {
        let names = cx.get_def_path(adt.did);

        if names.is_empty() {
            return false;
        }
        if names[0] == sym::libc || names[0] == sym::core && *names.last().unwrap() == sym!(c_void) {
            return true;
        }
    }
    false
}
