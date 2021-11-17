use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet;
use clippy_utils::ty::is_type_diagnostic_item;
use clippy_utils::{is_lang_ctor, single_segment_path};
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_hir::LangItem::{OptionNone, OptionSome};
use rustc_lint::LateContext;
use rustc_span::symbol::sym;

use super::OPTION_MAP_OR_NONE;
use super::RESULT_MAP_OR_INTO_OPTION;

// The expression inside a closure may or may not have surrounding braces
// which causes problems when generating a suggestion.
fn reduce_unit_expression<'a>(
    cx: &LateContext<'_>,
    expr: &'a hir::Expr<'_>,
) -> Option<(&'a hir::Expr<'a>, &'a [hir::Expr<'a>])> {
    match expr.kind {
        hir::ExprKind::Call(func, arg_char) => Some((func, arg_char)),
        hir::ExprKind::Block(block, _) => {
            match (block.stmts, block.expr) {
                (&[], Some(inner_expr)) => {
                    // If block only contains an expression,
                    // reduce `|x| { x + 1 }` to `|x| x + 1`
                    reduce_unit_expression(cx, inner_expr)
                },
                _ => None,
            }
        },
        _ => None,
    }
}

/// lint use of `_.map_or(None, _)` for `Option`s and `Result`s
pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx hir::Expr<'_>,
    recv: &'tcx hir::Expr<'_>,
    def_arg: &'tcx hir::Expr<'_>,
    map_arg: &'tcx hir::Expr<'_>,
) {
    let is_option = is_type_diagnostic_item(cx, cx.typeck_results().expr_ty(recv), sym::Option);
    let is_result = is_type_diagnostic_item(cx, cx.typeck_results().expr_ty(recv), sym::Result);

    // There are two variants of this `map_or` lint:
    // (1) using `map_or` as an adapter from `Result<T,E>` to `Option<T>`
    // (2) using `map_or` as a combinator instead of `and_then`
    //
    // (For this lint) we don't care if any other type calls `map_or`
    if !is_option && !is_result {
        return;
    }

    let default_arg_is_none = if let hir::ExprKind::Path(ref qpath) = def_arg.kind {
        is_lang_ctor(cx, qpath, OptionNone)
    } else {
        return;
    };

    if !default_arg_is_none {
        // nothing to lint!
        return;
    }

    let f_arg_is_some = if let hir::ExprKind::Path(ref qpath) = map_arg.kind {
        is_lang_ctor(cx, qpath, OptionSome)
    } else {
        false
    };

    if is_option {
        let self_snippet = snippet(cx, recv.span, "..");
        if_chain! {
        if let hir::ExprKind::Closure(_, _, id, span, _) = map_arg.kind;
            let arg_snippet = snippet(cx, span, "..");
            let body = cx.tcx.hir().body(id);
                if let Some((func, arg_char)) = reduce_unit_expression(cx, &body.value);
                if arg_char.len() == 1;
                if let hir::ExprKind::Path(ref qpath) = func.kind;
                if let Some(segment) = single_segment_path(qpath);
                if segment.ident.name == sym::Some;
                then {
                    let func_snippet = snippet(cx, arg_char[0].span, "..");
                    let msg = "called `map_or(None, ..)` on an `Option` value. This can be done more directly by calling \
                       `map(..)` instead";
                    return span_lint_and_sugg(
                        cx,
                        OPTION_MAP_OR_NONE,
                        expr.span,
                        msg,
                        "try using `map` instead",
                        format!("{0}.map({1} {2})", self_snippet, arg_snippet,func_snippet),
                        Applicability::MachineApplicable,
                    );
                }

        }

        let func_snippet = snippet(cx, map_arg.span, "..");
        let msg = "called `map_or(None, ..)` on an `Option` value. This can be done more directly by calling \
                       `and_then(..)` instead";
        return span_lint_and_sugg(
            cx,
            OPTION_MAP_OR_NONE,
            expr.span,
            msg,
            "try using `and_then` instead",
            format!("{0}.and_then({1})", self_snippet, func_snippet),
            Applicability::MachineApplicable,
        );
    } else if f_arg_is_some {
        let msg = "called `map_or(None, Some)` on a `Result` value. This can be done more directly by calling \
                       `ok()` instead";
        let self_snippet = snippet(cx, recv.span, "..");
        return span_lint_and_sugg(
            cx,
            RESULT_MAP_OR_INTO_OPTION,
            expr.span,
            msg,
            "try using `ok` instead",
            format!("{0}.ok()", self_snippet),
            Applicability::MachineApplicable,
        );
    }
}
