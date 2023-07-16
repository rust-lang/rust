use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::is_trait_method;
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::ty::is_type_diagnostic_item;
use rustc_errors::Applicability;
use rustc_hir::Expr;
use rustc_lint::LateContext;
use rustc_middle::ty;
use rustc_span::{symbol::sym, Span};

use super::MAP_FLATTEN;

/// lint use of `map().flatten()` for `Iterators` and 'Options'
pub(super) fn check(cx: &LateContext<'_>, expr: &Expr<'_>, recv: &Expr<'_>, map_arg: &Expr<'_>, map_span: Span) {
    if let Some((caller_ty_name, method_to_use)) = try_get_caller_ty_name_and_method_name(cx, expr, recv, map_arg) {
        let mut applicability = Applicability::MachineApplicable;

        let closure_snippet = snippet_with_applicability(cx, map_arg.span, "..", &mut applicability);
        span_lint_and_sugg(
            cx,
            MAP_FLATTEN,
            expr.span.with_lo(map_span.lo()),
            &format!("called `map(..).flatten()` on `{caller_ty_name}`"),
            &format!("try replacing `map` with `{method_to_use}` and remove the `.flatten()`"),
            format!("{method_to_use}({closure_snippet})"),
            applicability,
        );
    }
}

fn try_get_caller_ty_name_and_method_name(
    cx: &LateContext<'_>,
    expr: &Expr<'_>,
    caller_expr: &Expr<'_>,
    map_arg: &Expr<'_>,
) -> Option<(&'static str, &'static str)> {
    if is_trait_method(cx, expr, sym::Iterator) {
        if is_map_to_option(cx, map_arg) {
            // `(...).map(...)` has type `impl Iterator<Item=Option<...>>
            Some(("Iterator", "filter_map"))
        } else {
            // `(...).map(...)` has type `impl Iterator<Item=impl Iterator<...>>
            Some(("Iterator", "flat_map"))
        }
    } else {
        if let ty::Adt(adt, _) = cx.typeck_results().expr_ty(caller_expr).kind() {
            if cx.tcx.is_diagnostic_item(sym::Option, adt.did()) {
                return Some(("Option", "and_then"));
            } else if cx.tcx.is_diagnostic_item(sym::Result, adt.did()) {
                return Some(("Result", "and_then"));
            }
        }
        None
    }
}

fn is_map_to_option(cx: &LateContext<'_>, map_arg: &Expr<'_>) -> bool {
    let map_closure_ty = cx.typeck_results().expr_ty(map_arg);
    match map_closure_ty.kind() {
        ty::Closure(_, _) | ty::FnDef(_, _) | ty::FnPtr(_) => {
            let map_closure_sig = match map_closure_ty.kind() {
                ty::Closure(_, args) => args.as_closure().sig(),
                _ => map_closure_ty.fn_sig(cx.tcx),
            };
            let map_closure_return_ty = cx.tcx.erase_late_bound_regions(map_closure_sig.output());
            is_type_diagnostic_item(cx, map_closure_return_ty, sym::Option)
        },
        _ => false,
    }
}
