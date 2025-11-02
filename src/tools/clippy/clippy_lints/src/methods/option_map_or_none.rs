use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::res::{MaybeDef, MaybeQPath};
use clippy_utils::source::snippet;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_hir::LangItem::{OptionNone, OptionSome};
use rustc_lint::LateContext;
use rustc_span::symbol::sym;

use super::{OPTION_MAP_OR_NONE, RESULT_MAP_OR_INTO_OPTION};

// The expression inside a closure may or may not have surrounding braces
// which causes problems when generating a suggestion.
fn reduce_unit_expression<'a>(expr: &'a hir::Expr<'_>) -> Option<(&'a hir::Expr<'a>, &'a [hir::Expr<'a>])> {
    match expr.kind {
        hir::ExprKind::Call(func, arg_char) => Some((func, arg_char)),
        hir::ExprKind::Block(block, _) => {
            match (block.stmts, block.expr) {
                (&[], Some(inner_expr)) => {
                    // If block only contains an expression,
                    // reduce `|x| { x + 1 }` to `|x| x + 1`
                    reduce_unit_expression(inner_expr)
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
    let is_option = cx.typeck_results().expr_ty(recv).is_diag_item(cx, sym::Option);
    let is_result = cx.typeck_results().expr_ty(recv).is_diag_item(cx, sym::Result);

    // There are two variants of this `map_or` lint:
    // (1) using `map_or` as an adapter from `Result<T,E>` to `Option<T>`
    // (2) using `map_or` as a combinator instead of `and_then`
    //
    // (For this lint) we don't care if any other type calls `map_or`
    if !is_option && !is_result {
        return;
    }

    if !def_arg.res(cx).ctor_parent(cx).is_lang_item(cx, OptionNone) {
        // nothing to lint!
        return;
    }

    let f_arg_is_some = map_arg.res(cx).ctor_parent(cx).is_lang_item(cx, OptionSome);

    if is_option {
        let self_snippet = snippet(cx, recv.span, "..");
        if let hir::ExprKind::Closure(&hir::Closure { body, fn_decl_span, .. }) = map_arg.kind
            && let arg_snippet = snippet(cx, fn_decl_span, "..")
            && let body = cx.tcx.hir_body(body)
            && let Some((func, [arg_char])) = reduce_unit_expression(body.value)
            && let Some(id) = func.res(cx).opt_def_id().map(|ctor_id| cx.tcx.parent(ctor_id))
            && Some(id) == cx.tcx.lang_items().option_some_variant()
        {
            let func_snippet = snippet(cx, arg_char.span, "..");
            let msg = "called `map_or(None, ..)` on an `Option` value";
            return span_lint_and_sugg(
                cx,
                OPTION_MAP_OR_NONE,
                expr.span,
                msg,
                "consider using `map`",
                format!("{self_snippet}.map({arg_snippet} {func_snippet})"),
                Applicability::MachineApplicable,
            );
        }

        let func_snippet = snippet(cx, map_arg.span, "..");
        let msg = "called `map_or(None, ..)` on an `Option` value";
        span_lint_and_sugg(
            cx,
            OPTION_MAP_OR_NONE,
            expr.span,
            msg,
            "consider using `and_then`",
            format!("{self_snippet}.and_then({func_snippet})"),
            Applicability::MachineApplicable,
        );
    } else if f_arg_is_some {
        let msg = "called `map_or(None, Some)` on a `Result` value";
        let self_snippet = snippet(cx, recv.span, "..");
        span_lint_and_sugg(
            cx,
            RESULT_MAP_OR_INTO_OPTION,
            expr.span,
            msg,
            "consider using `ok`",
            format!("{self_snippet}.ok()"),
            Applicability::MachineApplicable,
        );
    }
}
