use rustc_errors::Applicability;
use rustc_hir::{
    intravisit::{walk_expr, NestedVisitorMap, Visitor},
    Expr, ExprKind,
};
use rustc_lint::LateContext;
use rustc_middle::hir::map::Map;
use rustc_span::source_map::Span;

use if_chain::if_chain;

use crate::utils::{has_iter_method, match_trait_method, paths, snippet, span_lint_and_then};

use super::EXCESSIVE_FOR_EACH;

pub(super) fn lint(cx: &LateContext<'_>, expr: &'tcx Expr<'_>, args: &[&[Expr<'_>]]) {
    if args.len() < 2 {
        return;
    }

    let for_each_args = args[0];
    if for_each_args.len() < 2 {
        return;
    }
    let for_each_receiver = &for_each_args[0];
    let for_each_arg = &for_each_args[1];
    let iter_receiver = &args[1][0];

    if_chain! {
        if has_iter_method(cx, cx.typeck_results().expr_ty(iter_receiver)).is_some();
        if match_trait_method(cx, expr, &paths::ITERATOR);
        if let ExprKind::Closure(_, _, body_id, ..) = for_each_arg.kind;
        let body = cx.tcx.hir().body(body_id);
        if let ExprKind::Block(..) = body.value.kind;
        then {
            let mut ret_collector = RetCollector::new();
            ret_collector.visit_expr(&body.value);

            // Skip the lint if `return` is used in `Loop` in order not to suggest using `'label`.
            if ret_collector.ret_in_loop {
                return;
            }

            let sugg = format!(
                "for {} in {} {{ .. }}",
                snippet(cx, body.params[0].pat.span, ".."),
                snippet(cx, for_each_receiver.span, "..")
            );

            span_lint_and_then(
                cx,
                EXCESSIVE_FOR_EACH,
                expr.span,
                "excessive use of `for_each`",
                |diag| {
                    diag.span_suggestion(expr.span, "try", sugg, Applicability::HasPlaceholders);
                    for span in ret_collector.spans {
                        diag.span_note(span, "change `return` to `continue` in the loop body");
                    }
                }
            )
        }
    }
}

/// This type plays two roles.
/// 1. Collect spans of `return` in the closure body.
/// 2. Detect use of `return` in `Loop` in the closure body.
///
/// NOTE: The functionality of this type is similar to
/// [`crate::utilts::visitors::find_all_ret_expressions`], but we can't use
/// `find_all_ret_expressions` instead of this type. The reasons are:
/// 1. `find_all_ret_expressions` passes the argument of `ExprKind::Ret` to a callback, but what we
///    need here is `ExprKind::Ret` itself.
/// 2. We can't trace current loop depth with `find_all_ret_expressions`.
struct RetCollector {
    spans: Vec<Span>,
    ret_in_loop: bool,

    loop_depth: u16,
}

impl RetCollector {
    fn new() -> Self {
        Self {
            spans: Vec::new(),
            ret_in_loop: false,
            loop_depth: 0,
        }
    }
}

impl<'tcx> Visitor<'tcx> for RetCollector {
    type Map = Map<'tcx>;

    fn visit_expr(&mut self, expr: &Expr<'_>) {
        match expr.kind {
            ExprKind::Ret(..) => {
                if self.loop_depth > 0 && !self.ret_in_loop {
                    self.ret_in_loop = true
                }

                self.spans.push(expr.span)
            },

            ExprKind::Loop(..) => {
                self.loop_depth += 1;
                walk_expr(self, expr);
                self.loop_depth -= 1;
                return;
            },

            _ => {},
        }

        walk_expr(self, expr);
    }

    fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
        NestedVisitorMap::None
    }
}
