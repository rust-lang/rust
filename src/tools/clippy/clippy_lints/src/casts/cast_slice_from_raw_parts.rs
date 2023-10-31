use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::source::snippet_with_context;
use clippy_utils::{match_def_path, paths};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::def_id::DefId;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::LateContext;
use rustc_middle::ty::{self, Ty};

use super::CAST_SLICE_FROM_RAW_PARTS;

enum RawPartsKind {
    Immutable,
    Mutable,
}

fn raw_parts_kind(cx: &LateContext<'_>, did: DefId) -> Option<RawPartsKind> {
    if match_def_path(cx, did, &paths::SLICE_FROM_RAW_PARTS) {
        Some(RawPartsKind::Immutable)
    } else if match_def_path(cx, did, &paths::SLICE_FROM_RAW_PARTS_MUT) {
        Some(RawPartsKind::Mutable)
    } else {
        None
    }
}

pub(super) fn check(cx: &LateContext<'_>, expr: &Expr<'_>, cast_expr: &Expr<'_>, cast_to: Ty<'_>, msrv: &Msrv) {
    if_chain! {
        if msrv.meets(msrvs::PTR_SLICE_RAW_PARTS);
        if let ty::RawPtr(ptrty) = cast_to.kind();
        if let ty::Slice(_) = ptrty.ty.kind();
        if let ExprKind::Call(fun, [ptr_arg, len_arg]) = cast_expr.peel_blocks().kind;
        if let ExprKind::Path(ref qpath) = fun.kind;
        if let Some(fun_def_id) = cx.qpath_res(qpath, fun.hir_id).opt_def_id();
        if let Some(rpk) = raw_parts_kind(cx, fun_def_id);
        let ctxt = expr.span.ctxt();
        if cast_expr.span.ctxt() == ctxt;
        then {
            let func = match rpk {
                RawPartsKind::Immutable => "from_raw_parts",
                RawPartsKind::Mutable => "from_raw_parts_mut"
            };
            let span = expr.span;
            let mut applicability = Applicability::MachineApplicable;
            let ptr = snippet_with_context(cx, ptr_arg.span, ctxt, "ptr", &mut applicability).0;
            let len = snippet_with_context(cx, len_arg.span, ctxt, "len", &mut applicability).0;
            span_lint_and_sugg(
                cx,
                CAST_SLICE_FROM_RAW_PARTS,
                span,
                &format!("casting the result of `{func}` to {cast_to}"),
                "replace with",
                format!("core::ptr::slice_{func}({ptr}, {len})"),
                applicability
            );
        }
    }
}
