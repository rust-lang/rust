use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_lint::LateContext;
use rustc_middle::ty::{self, GenericArg, Ty};
use rustc_span::sym;
use std::ops::ControlFlow;

use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::{indent_of, reindent_multiline, snippet_with_applicability};
use clippy_utils::ty::get_type_diagnostic_name;
use clippy_utils::visitors::for_each_unconsumed_temporary;
use clippy_utils::{is_expr_final_block_expr, peel_blocks};

use super::RETURN_AND_THEN;

/// lint if `and_then` is the last expression in a block, and
/// there are no references or temporaries in the receiver
pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &hir::Expr<'_>,
    recv: &'tcx hir::Expr<'tcx>,
    arg: &'tcx hir::Expr<'_>,
) {
    if !is_expr_final_block_expr(cx.tcx, expr) {
        return;
    }

    let recv_type = cx.typeck_results().expr_ty(recv);
    if !matches!(get_type_diagnostic_name(cx, recv_type), Some(sym::Option | sym::Result)) {
        return;
    }

    let has_ref_type = matches!(recv_type.kind(), ty::Adt(_, args) if args
        .first()
        .and_then(|arg0: &GenericArg<'tcx>| GenericArg::as_type(*arg0))
        .is_some_and(Ty::is_ref));
    let has_temporaries = for_each_unconsumed_temporary(cx, recv, |_| ControlFlow::Break(())).is_break();
    if has_ref_type && has_temporaries {
        return;
    }

    let hir::ExprKind::Closure(&hir::Closure { body, fn_decl, .. }) = arg.kind else {
        return;
    };

    let closure_arg = fn_decl.inputs[0];
    let closure_expr = peel_blocks(cx.tcx.hir_body(body).value);

    let mut applicability = Applicability::MachineApplicable;
    let arg_snip = snippet_with_applicability(cx, closure_arg.span, "_", &mut applicability);
    let recv_snip = snippet_with_applicability(cx, recv.span, "_", &mut applicability);
    let body_snip = snippet_with_applicability(cx, closure_expr.span, "..", &mut applicability);
    let inner = match body_snip.strip_prefix('{').and_then(|s| s.strip_suffix('}')) {
        Some(s) => s.trim_start_matches('\n').trim_end(),
        None => &body_snip,
    };

    let msg = "use the question mark operator instead of an `and_then` call";
    let sugg = format!(
        "let {} = {}?;\n{}",
        arg_snip,
        recv_snip,
        reindent_multiline(inner.into(), false, indent_of(cx, expr.span))
    );

    span_lint_and_sugg(cx, RETURN_AND_THEN, expr.span, msg, "try", sugg, applicability);
}
