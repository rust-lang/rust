use rustc_errors::Applicability;
use rustc_hir::{
    intravisit::{walk_expr, NestedVisitorMap, Visitor},
    Expr, ExprKind, Stmt, StmtKind,
};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::hir::map::Map;
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::{source_map::Span, sym, Symbol};

use if_chain::if_chain;

use crate::utils::{
    has_iter_method, is_diagnostic_assoc_item, method_calls, snippet_with_applicability, span_lint_and_then,
};

declare_clippy_lint! {
    /// **What it does:** Checks for usage of `for_each` that would be more simply written as a
    /// `for` loop.
    ///
    /// **Why is this bad?** `for_each` may be used after applying iterator transformers like
    /// `filter` for better readability and performance. It may also be used to fit a simple
    /// operation on one line.
    /// But when none of these apply, a simple `for` loop is more idiomatic.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    ///
    /// ```rust
    /// let v = vec![0, 1, 2];
    /// v.iter().for_each(|elem| {
    ///     println!("{}", elem);
    /// })
    /// ```
    /// Use instead:
    /// ```rust
    /// let v = vec![0, 1, 2];
    /// for elem in v.iter() {
    ///     println!("{}", elem);
    /// }
    /// ```
    pub NEEDLESS_FOR_EACH,
    restriction,
    "using `for_each` where a `for` loop would be simpler"
}

declare_lint_pass!(NeedlessForEach => [NEEDLESS_FOR_EACH]);

impl LateLintPass<'_> for NeedlessForEach {
    fn check_stmt(&mut self, cx: &LateContext<'tcx>, stmt: &'tcx Stmt<'_>) {
        let expr = match stmt.kind {
            StmtKind::Expr(expr) | StmtKind::Semi(expr) => expr,
            StmtKind::Local(local) if local.init.is_some() => local.init.unwrap(),
            _ => return,
        };

        // Max depth is set to 3 because we need to check the method chain length is just two.
        let (method_names, arg_lists, _) = method_calls(expr, 3);

        if_chain! {
            // This assures the length of this method chain is two.
            if let [for_each_args, iter_args] = arg_lists.as_slice();
            if let Some(for_each_sym) = method_names.first();
            if *for_each_sym == Symbol::intern("for_each");
            if let Some(did) = cx.typeck_results().type_dependent_def_id(expr.hir_id);
            if is_diagnostic_assoc_item(cx, did, sym::Iterator);
            // Checks the type of the first method receiver is NOT a user defined type.
            if has_iter_method(cx, cx.typeck_results().expr_ty(&iter_args[0])).is_some();
            if let ExprKind::Closure(_, _, body_id, ..) = for_each_args[1].kind;
            let body = cx.tcx.hir().body(body_id);
            // Skip the lint if the body is not block because this is simpler than `for` loop.
            // e.g. `v.iter().for_each(f)` is simpler and clearer than using `for` loop.
            if let ExprKind::Block(..) = body.value.kind;
            then {
                let mut ret_collector = RetCollector::default();
                ret_collector.visit_expr(&body.value);

                // Skip the lint if `return` is used in `Loop` in order not to suggest using `'label`.
                if ret_collector.ret_in_loop {
                    return;
                }

                // We can't use `Applicability::MachineApplicable` when the closure contains `return`
                // because `Diagnostic::multipart_suggestion` doesn't work with multiple overlapped
                // spans.
                let mut applicability = if ret_collector.spans.is_empty() {
                    Applicability::MachineApplicable
                } else {
                    Applicability::MaybeIncorrect
                };

                let mut suggs = vec![];
                suggs.push((stmt.span, format!(
                    "for {} in {} {}",
                    snippet_with_applicability(cx, body.params[0].pat.span, "..", &mut applicability),
                    snippet_with_applicability(cx, for_each_args[0].span, "..", &mut applicability),
                    snippet_with_applicability(cx, body.value.span, "..", &mut applicability),
                )));

                for span in &ret_collector.spans {
                    suggs.push((*span, "return".to_string()));
                }

                span_lint_and_then(
                    cx,
                    NEEDLESS_FOR_EACH,
                    stmt.span,
                    "needless use of `for_each`",
                    |diag| {
                        diag.multipart_suggestion("try", suggs, applicability);
                        // `Diagnostic::multipart_suggestion` ignores the second and subsequent overlapped spans,
                        // so `span_note` is needed here even though `suggs` includes the replacements.
                        for span in ret_collector.spans {
                            diag.span_note(span, "replace `return` with `continue`");
                        }
                    }
                )
            }
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
#[derive(Default)]
struct RetCollector {
    spans: Vec<Span>,
    ret_in_loop: bool,
    loop_depth: u16,
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
