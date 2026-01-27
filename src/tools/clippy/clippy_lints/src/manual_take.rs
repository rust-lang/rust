use clippy_config::Conf;
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::msrvs::{MEM_TAKE, Msrv};
use clippy_utils::source::snippet_with_context;
use rustc_ast::LitKind;
use rustc_errors::Applicability;
use rustc_hir::{Block, Expr, ExprKind, StmtKind};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_session::impl_lint_pass;

declare_clippy_lint! {
    /// ### What it does
    /// Detects manual re-implementations of `std::mem::take`.
    ///
    /// ### Why is this bad?
    /// Because the function call is shorter and easier to read.
    ///
    /// ### Known issues
    /// Currently the lint only detects cases involving `bool`s.
    ///
    /// ### Example
    /// ```no_run
    /// let mut x = true;
    /// let _ = if x {
    ///     x = false;
    ///     true
    /// } else {
    ///     false
    /// };
    /// ```
    /// Use instead:
    /// ```no_run
    /// let mut x = true;
    /// let _ = std::mem::take(&mut x);
    /// ```
    #[clippy::version = "1.94.0"]
    pub MANUAL_TAKE,
    complexity,
    "manual `mem::take` implementation"
}
pub struct ManualTake {
    msrv: Msrv,
}

impl ManualTake {
    pub fn new(conf: &'static Conf) -> Self {
        Self { msrv: conf.msrv }
    }
}

impl_lint_pass!(ManualTake => [MANUAL_TAKE]);

impl LateLintPass<'_> for ManualTake {
    fn check_expr(&mut self, cx: &LateContext<'_>, expr: &Expr<'_>) {
        if let ExprKind::If(cond, then, Some(otherwise)) = expr.kind
            && let ExprKind::Path(_) = cond.kind
            && let ExprKind::Block(
                Block {
                    stmts: [stmt],
                    expr: Some(then_expr),
                    ..
                },
                ..,
            ) = then.kind
            && let ExprKind::Block(
                Block {
                    stmts: [],
                    expr: Some(else_expr),
                    ..
                },
                ..,
            ) = otherwise.kind
            && let StmtKind::Semi(assignment) = stmt.kind
            && let ExprKind::Assign(mut_c, possible_false, _) = assignment.kind
            && let ExprKind::Path(_) = mut_c.kind
            && !expr.span.in_external_macro(cx.sess().source_map())
            && let Some(std_or_core) = clippy_utils::std_or_core(cx)
            && self.msrv.meets(cx, MEM_TAKE)
            && clippy_utils::SpanlessEq::new(cx).eq_expr(cond, mut_c)
            && Some(false) == as_const_bool(possible_false)
            && let Some(then_bool) = as_const_bool(then_expr)
            && let Some(else_bool) = as_const_bool(else_expr)
            && then_bool != else_bool
        {
            span_lint_and_then(
                cx,
                MANUAL_TAKE,
                expr.span,
                "manual implementation of `mem::take`",
                |diag| {
                    let mut app = Applicability::MachineApplicable;
                    let negate = if then_bool { "" } else { "!" };
                    let taken = snippet_with_context(cx, cond.span, expr.span.ctxt(), "_", &mut app).0;
                    diag.span_suggestion_verbose(
                        expr.span,
                        "use",
                        format!("{negate}{std_or_core}::mem::take(&mut {taken})"),
                        app,
                    );
                },
            );
        }
    }
}

fn as_const_bool(e: &Expr<'_>) -> Option<bool> {
    if let ExprKind::Lit(lit) = e.kind
        && let LitKind::Bool(b) = lit.node
    {
        Some(b)
    } else {
        None
    }
}
