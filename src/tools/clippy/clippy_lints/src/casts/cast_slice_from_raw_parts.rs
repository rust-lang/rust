use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::source::snippet_with_context;
use clippy_utils::{get_parent_expr, is_no_std_crate};
use rustc_errors::Applicability;
use rustc_hir::def_id::DefId;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::LateContext;
use rustc_middle::ty::adjustment::{Adjust, Adjustment, AutoBorrow};
use rustc_middle::ty::{self, Ty};
use rustc_span::sym;

use super::CAST_SLICE_FROM_RAW_PARTS;

enum RawPartsKind {
    Immutable,
    Mutable,
}

fn raw_parts_kind(cx: &LateContext<'_>, did: DefId) -> Option<RawPartsKind> {
    match cx.tcx.get_diagnostic_name(did)? {
        sym::slice_from_raw_parts => Some(RawPartsKind::Immutable),
        sym::slice_from_raw_parts_mut => Some(RawPartsKind::Mutable),
        _ => None,
    }
}

pub(super) fn check(cx: &LateContext<'_>, expr: &Expr<'_>, cast_expr: &Expr<'_>, cast_to: Ty<'_>, msrv: Msrv) {
    if let ty::RawPtr(ptrty, _) = cast_to.kind()
        && let ty::Slice(_) = ptrty.kind()
        && let ExprKind::Call(fun, [ptr_arg, len_arg]) = cast_expr.peel_blocks().kind
        && let ExprKind::Path(ref qpath) = fun.kind
        && let Some(fun_def_id) = cx.qpath_res(qpath, fun.hir_id).opt_def_id()
        && let Some(rpk) = raw_parts_kind(cx, fun_def_id)
        && let ctxt = expr.span.ctxt()
        && cast_expr.span.ctxt() == ctxt
        && msrv.meets(cx, msrvs::PTR_SLICE_RAW_PARTS)
    {
        let func = match rpk {
            RawPartsKind::Immutable => "from_raw_parts",
            RawPartsKind::Mutable => "from_raw_parts_mut",
        };
        let span = expr.span;
        let mut applicability = Applicability::MachineApplicable;
        let ptr = snippet_with_context(cx, ptr_arg.span, ctxt, "ptr", &mut applicability).0;
        let len = snippet_with_context(cx, len_arg.span, ctxt, "len", &mut applicability).0;
        let krate = if is_no_std_crate(cx) { "core" } else { "std" };
        span_lint_and_sugg(
            cx,
            CAST_SLICE_FROM_RAW_PARTS,
            span,
            format!("casting the result of `{func}` to {cast_to}"),
            "replace with",
            format!("{krate}::ptr::slice_{func}({ptr}, {len})"),
            applicability,
        );
    }
}

/// Checks for implicit cast from slice reference to raw slice pointer.
pub(super) fn check_implicit_cast(cx: &LateContext<'_>, expr: &Expr<'_>) {
    if let ExprKind::Call(fun, [ptr_arg, len_arg]) = expr.peel_blocks().kind
        && let ExprKind::Path(ref qpath) = fun.kind
        && let Some(fun_def_id) = cx.qpath_res(qpath, fun.hir_id).opt_def_id()
        && let Some(rpk) = raw_parts_kind(cx, fun_def_id)
        && !matches!(get_parent_expr(cx, expr).map(|e| e.kind), Some(ExprKind::Cast(..)))
        && let [deref, borrow] = cx.typeck_results().expr_adjustments(expr)
        && matches!(deref.kind, Adjust::Deref(..))
        && let Adjustment {
            kind: Adjust::Borrow(AutoBorrow::RawPtr(..)),
            target,
        } = borrow
        && let ty::RawPtr(pointee_ty, _) = target.kind()
        && pointee_ty.is_slice()
        && !expr.span.from_expansion()
    {
        let func = match rpk {
            RawPartsKind::Immutable => "from_raw_parts",
            RawPartsKind::Mutable => "from_raw_parts_mut",
        };
        let mut applicability = Applicability::MachineApplicable;
        let ctxt = expr.span.ctxt();
        let ptr = snippet_with_context(cx, ptr_arg.span, ctxt, "ptr", &mut applicability).0;
        let len = snippet_with_context(cx, len_arg.span, ctxt, "len", &mut applicability).0;
        let krate = if is_no_std_crate(cx) { "core" } else { "std" };
        span_lint_and_sugg(
            cx,
            CAST_SLICE_FROM_RAW_PARTS,
            expr.span,
            format!("implicitly casting the result of `{func}` to `{target}`"),
            "replace_with",
            format!("{krate}::ptr::slice_{func}({ptr}, {len})"),
            applicability,
        );
    }
}
