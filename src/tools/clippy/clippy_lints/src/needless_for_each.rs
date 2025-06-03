use rustc_errors::Applicability;
use rustc_hir::intravisit::{Visitor, walk_expr};
use rustc_hir::{Block, BlockCheckMode, Closure, Expr, ExprKind, Stmt, StmtKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::declare_lint_pass;
use rustc_span::Span;

use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::ty::has_iter_method;
use clippy_utils::{is_trait_method, sym};

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
    /// ```no_run
    /// let v = vec![0, 1, 2];
    /// v.iter().for_each(|elem| {
    ///     println!("{elem}");
    /// })
    /// ```
    /// Use instead:
    /// ```no_run
    /// let v = vec![0, 1, 2];
    /// for elem in &v {
    ///     println!("{elem}");
    /// }
    /// ```
    ///
    /// ### Known Problems
    /// When doing things such as:
    /// ```ignore
    /// let v = vec![0, 1, 2];
    /// v.iter().for_each(|elem| unsafe {
    ///     libc::printf(c"%d\n".as_ptr(), elem);
    /// });
    /// ```
    /// This lint will not trigger.
    #[clippy::version = "1.53.0"]
    pub NEEDLESS_FOR_EACH,
    pedantic,
    "using `for_each` where a `for` loop would be simpler"
}

declare_lint_pass!(NeedlessForEach => [NEEDLESS_FOR_EACH]);

impl<'tcx> LateLintPass<'tcx> for NeedlessForEach {
    fn check_stmt(&mut self, cx: &LateContext<'tcx>, stmt: &'tcx Stmt<'_>) {
        if let StmtKind::Expr(expr) | StmtKind::Semi(expr) = stmt.kind
            && let ExprKind::MethodCall(method_name, for_each_recv, [for_each_arg], _) = expr.kind
            && let ExprKind::MethodCall(_, iter_recv, [], _) = for_each_recv.kind
            // Skip the lint if the call chain is too long. e.g. `v.field.iter().for_each()` or
            // `v.foo().iter().for_each()` must be skipped.
            && matches!(
                iter_recv.kind,
                ExprKind::Array(..) | ExprKind::Call(..) | ExprKind::Path(..)
            )
            && method_name.ident.name == sym::for_each
            && is_trait_method(cx, expr, sym::Iterator)
            // Checks the type of the `iter` method receiver is NOT a user defined type.
            && has_iter_method(cx, cx.typeck_results().expr_ty(iter_recv)).is_some()
            // Skip the lint if the body is not block because this is simpler than `for` loop.
            // e.g. `v.iter().for_each(f)` is simpler and clearer than using `for` loop.
            && let ExprKind::Closure(&Closure { body, .. }) = for_each_arg.kind
            && let body = cx.tcx.hir_body(body)
            // Skip the lint if the body is not safe, so as not to suggest `for … in … unsafe {}`
            // and suggesting `for … in … { unsafe { } }` is a little ugly.
            && !matches!(body.value.kind, ExprKind::Block(Block { rules: BlockCheckMode::UnsafeBlock(_), .. }, ..))
        {
            let mut ret_collector = RetCollector::default();
            ret_collector.visit_expr(body.value);

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

            let body_param_sugg = snippet_with_applicability(cx, body.params[0].pat.span, "..", &mut applicability);
            let for_each_rev_sugg = snippet_with_applicability(cx, for_each_recv.span, "..", &mut applicability);
            let body_value_sugg = snippet_with_applicability(cx, body.value.span, "..", &mut applicability);

            let sugg = format!(
                "for {} in {} {}",
                body_param_sugg,
                for_each_rev_sugg,
                match body.value.kind {
                    ExprKind::Block(block, _) if is_let_desugar(block) => {
                        format!("{{ {body_value_sugg} }}")
                    },
                    ExprKind::Block(_, _) => body_value_sugg.to_string(),
                    _ => format!("{{ {body_value_sugg}; }}"),
                }
            );

            span_lint_and_then(cx, NEEDLESS_FOR_EACH, stmt.span, "needless use of `for_each`", |diag| {
                diag.span_suggestion(stmt.span, "try", sugg, applicability);
                if let Some(ret_suggs) = ret_suggs {
                    diag.multipart_suggestion("...and replace `return` with `continue`", ret_suggs, applicability);
                }
            });
        }
    }
}

/// Check if the block is a desugared `_ = expr` statement.
fn is_let_desugar(block: &Block<'_>) -> bool {
    matches!(
        block,
        Block {
            stmts: [Stmt {
                kind: StmtKind::Let(_),
                ..
            },],
            ..
        }
    )
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

impl Visitor<'_> for RetCollector {
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
}
