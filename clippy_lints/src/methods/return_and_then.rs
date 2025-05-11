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
use clippy_utils::{get_parent_expr, peel_blocks};

use super::RETURN_AND_THEN;

/// lint if `and_then` is the last expression in a block, and
/// there are no references or temporaries in the receiver
pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &hir::Expr<'_>,
    recv: &'tcx hir::Expr<'tcx>,
    arg: &'tcx hir::Expr<'_>,
) {
    if cx.tcx.hir_get_fn_id_for_return_block(expr.hir_id).is_none() {
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

    // If suggestion is going to get inserted as part of a `return` expression, it must be blockified.
    let sugg = if let Some(parent_expr) = get_parent_expr(cx, expr) {
        let base_indent = indent_of(cx, parent_expr.span);
        let inner_indent = base_indent.map(|i| i + 4);
        format!(
            "{}\n{}\n{}",
            reindent_multiline(&format!("{{\nlet {arg_snip} = {recv_snip}?;"), true, inner_indent),
            reindent_multiline(inner, false, inner_indent),
            reindent_multiline("}", false, base_indent),
        )
    } else {
        format!(
            "let {} = {}?;\n{}",
            arg_snip,
            recv_snip,
            reindent_multiline(inner, false, indent_of(cx, expr.span))
        )
    };

    span_lint_and_sugg(
        cx,
        RETURN_AND_THEN,
        expr.span,
        "use the `?` operator instead of an `and_then` call",
        "try",
        sugg,
        applicability,
    );
}
