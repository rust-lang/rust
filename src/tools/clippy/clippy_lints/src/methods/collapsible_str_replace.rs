use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet;
use clippy_utils::visitors::for_each_expr;
use clippy_utils::{eq_expr_value, get_parent_expr};
use core::ops::ControlFlow;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_lint::LateContext;
use std::collections::VecDeque;

use super::method_call;
use super::COLLAPSIBLE_STR_REPLACE;

pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx hir::Expr<'tcx>,
    from: &'tcx hir::Expr<'tcx>,
    to: &'tcx hir::Expr<'tcx>,
) {
    let replace_methods = collect_replace_calls(cx, expr, to);
    if replace_methods.methods.len() > 1 {
        let from_kind = cx.typeck_results().expr_ty(from).peel_refs().kind();
        // If the parent node's `to` argument is the same as the `to` argument
        // of the last replace call in the current chain, don't lint as it was already linted
        if let Some(parent) = get_parent_expr(cx, expr)
            && let Some(("replace", _, [current_from, current_to], _, _)) = method_call(parent)
            && eq_expr_value(cx, to, current_to)
            && from_kind == cx.typeck_results().expr_ty(current_from).peel_refs().kind()
        {
            return;
        }

        check_consecutive_replace_calls(cx, expr, &replace_methods, to);
    }
}

struct ReplaceMethods<'tcx> {
    methods: VecDeque<&'tcx hir::Expr<'tcx>>,
    from_args: VecDeque<&'tcx hir::Expr<'tcx>>,
}

fn collect_replace_calls<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx hir::Expr<'tcx>,
    to_arg: &'tcx hir::Expr<'tcx>,
) -> ReplaceMethods<'tcx> {
    let mut methods = VecDeque::new();
    let mut from_args = VecDeque::new();

    let _: Option<()> = for_each_expr(expr, |e| {
        if let Some(("replace", _, [from, to], _, _)) = method_call(e) {
            if eq_expr_value(cx, to_arg, to) && cx.typeck_results().expr_ty(from).peel_refs().is_char() {
                methods.push_front(e);
                from_args.push_front(from);
                ControlFlow::Continue(())
            } else {
                ControlFlow::Break(())
            }
        } else {
            ControlFlow::Continue(())
        }
    });

    ReplaceMethods { methods, from_args }
}

/// Check a chain of `str::replace` calls for `collapsible_str_replace` lint.
fn check_consecutive_replace_calls<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx hir::Expr<'tcx>,
    replace_methods: &ReplaceMethods<'tcx>,
    to_arg: &'tcx hir::Expr<'tcx>,
) {
    let from_args = &replace_methods.from_args;
    let from_arg_reprs: Vec<String> = from_args
        .iter()
        .map(|from_arg| snippet(cx, from_arg.span, "..").to_string())
        .collect();
    let app = Applicability::MachineApplicable;
    let earliest_replace_call = replace_methods.methods.front().unwrap();
    if let Some((_, _, [..], span_lo, _)) = method_call(earliest_replace_call) {
        span_lint_and_sugg(
            cx,
            COLLAPSIBLE_STR_REPLACE,
            expr.span.with_lo(span_lo.lo()),
            "used consecutive `str::replace` call",
            "replace with",
            format!(
                "replace([{}], {})",
                from_arg_reprs.join(", "),
                snippet(cx, to_arg.span, ".."),
            ),
            app,
        );
    }
}
