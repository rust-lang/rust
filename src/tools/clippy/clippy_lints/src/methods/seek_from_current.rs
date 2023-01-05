use rustc_ast::ast::{LitIntType, LitKind};
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::LateContext;

use clippy_utils::{
    diagnostics::span_lint_and_sugg, get_trait_def_id, match_def_path, paths, source::snippet_with_applicability,
    ty::implements_trait,
};

use super::SEEK_FROM_CURRENT;

pub(super) fn check<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>, recv: &'tcx Expr<'_>, arg: &'tcx Expr<'_>) {
    let ty = cx.typeck_results().expr_ty(recv);

    if let Some(def_id) = get_trait_def_id(cx, &paths::STD_IO_SEEK) {
        if implements_trait(cx, ty, def_id, &[]) && arg_is_seek_from_current(cx, arg) {
            let mut applicability = Applicability::MachineApplicable;
            let snip = snippet_with_applicability(cx, recv.span, "..", &mut applicability);

            span_lint_and_sugg(
                cx,
                SEEK_FROM_CURRENT,
                expr.span,
                "using `SeekFrom::Current` to start from current position",
                "replace with",
                format!("{snip}.stream_position()"),
                applicability,
            );
        }
    }
}

fn arg_is_seek_from_current<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) -> bool {
    if let ExprKind::Call(f, args) = expr.kind &&
        let ExprKind::Path(ref path) = f.kind &&
        let Some(def_id) = cx.qpath_res(path, f.hir_id).opt_def_id() &&
        match_def_path(cx, def_id, &paths::STD_IO_SEEK_FROM_CURRENT) {
        // check if argument of `SeekFrom::Current` is `0`
        if args.len() == 1 &&
            let ExprKind::Lit(ref lit) = args[0].kind &&
            let LitKind::Int(0, LitIntType::Unsuffixed) = lit.node {
            return true
        }
    }

    false
}
