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

use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::is_trait_method;
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::ty::has_iter_method;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `for_each` that would be more simply written as a
    /// `for` loop.
    ///
    /// ### Why is this bad?
    /// `for_each` may be used after applying iterator transformers like
    /// `filter` for better readability and performance. It may also be used to fit a simple
    /// operation on one line.
    /// But when none of these apply, a simple `for` loop is more idiomatic.
    ///
    /// ### Example
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
    #[clippy::version = "1.53.0"]
    pub NEEDLESS_FOR_EACH,
    pedantic,
    "using `for_each` where a `for` loop would be simpler"
}

declare_lint_pass!(NeedlessForEach => [NEEDLESS_FOR_EACH]);

impl LateLintPass<'_> for NeedlessForEach {
    fn check_stmt(&mut self, cx: &LateContext<'tcx>, stmt: &'tcx Stmt<'_>) {
        let expr = match stmt.kind {
            StmtKind::Expr(expr) | StmtKind::Semi(expr) => expr,
            _ => return,
        };

        if_chain! {
            // Check the method name is `for_each`.
            if let ExprKind::MethodCall(method_name, _, [for_each_recv, for_each_arg], _) = expr.kind;
            if method_name.ident.name == Symbol::intern("for_each");
            // Check `for_each` is an associated function of `Iterator`.
            if is_trait_method(cx, expr, sym::Iterator);
            // Checks the receiver of `for_each` is also a method call.
            if let ExprKind::MethodCall(_, _, [iter_recv], _) = for_each_recv.kind;
            // Skip the lint if the call chain is too long. e.g. `v.field.iter().for_each()` or
            // `v.foo().iter().for_each()` must be skipped.
            if matches!(
                iter_recv.kind,
                ExprKind::Array(..) | ExprKind::Call(..) | ExprKind::Path(..)
            );
            // Checks the type of the `iter` method receiver is NOT a user defined type.
            if has_iter_method(cx, cx.typeck_results().expr_ty(iter_recv)).is_some();
            // Skip the lint if the body is not block because this is simpler than `for` loop.
            // e.g. `v.iter().for_each(f)` is simpler and clearer than using `for` loop.
            if let ExprKind::Closure(_, _, body_id, ..) = for_each_arg.kind;
            let body = cx.tcx.hir().body(body_id);
            if let ExprKind::Block(..) = body.value.kind;
            then {
                let mut ret_collector = RetCollector::default();
                ret_collector.visit_expr(&body.value);

                // Skip the lint if `return` is used in `Loop` in order not to suggest using `'label`.
                if ret_collector.ret_in_loop {
                    return;
                }

                let (mut applicability, ret_suggs) = if ret_collector.spans.is_empty() {
                    (Applicability::MachineApplicable, None)
                } else {
                    (
                        Applicability::MaybeIncorrect,
                        Some(
                            ret_collector
                                .spans
                                .into_iter()
                                .map(|span| (span, "continue".to_string()))
                                .collect(),
                        ),
                    )
                };

                let sugg = format!(
                    "for {} in {} {}",
                    snippet_with_applicability(cx, body.params[0].pat.span, "..", &mut applicability),
                    snippet_with_applicability(cx, for_each_recv.span, "..", &mut applicability),
                    snippet_with_applicability(cx, body.value.span, "..", &mut applicability),
                );

                span_lint_and_then(cx, NEEDLESS_FOR_EACH, stmt.span, "needless use of `for_each`", |diag| {
                    diag.span_suggestion(stmt.span, "try", sugg, applicability);
                    if let Some(ret_suggs) = ret_suggs {
                        diag.multipart_suggestion("...and replace `return` with `continue`", ret_suggs, applicability);
                    }
                })
            }
        }
    }
}

/// This type plays two roles.
/// 1. Collect spans of `return` in the closure body.
/// 2. Detect use of `return` in `Loop` in the closure body.
///
/// NOTE: The functionality of this type is similar to
/// [`clippy_utils::visitors::find_all_ret_expressions`], but we can't use
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
                    self.ret_in_loop = true;
                }

                self.spans.push(expr.span);
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
