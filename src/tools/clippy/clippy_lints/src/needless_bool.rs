use clippy_utils::diagnostics::{span_lint, span_lint_and_sugg};
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::sugg::Sugg;
use clippy_utils::{
    SpanlessEq, get_parent_expr, higher, is_block_like, is_else_clause, is_parent_stmt, is_receiver_of_method_call,
    peel_blocks, peel_blocks_with_stmt, span_contains_comment,
};
use rustc_ast::ast::LitKind;
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::declare_lint_pass;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for expressions of the form `if c { true } else {
    /// false }` (or vice versa) and suggests using the condition directly.
    ///
    /// ### Why is this bad?
    /// Redundant code.
    ///
    /// ### Known problems
    /// Maybe false positives: Sometimes, the two branches are
    /// painstakingly documented (which we, of course, do not detect), so they *may*
    /// have some value. Even then, the documentation can be rewritten to match the
    /// shorter code.
    ///
    /// ### Example
    /// ```no_run
    /// # let x = true;
    /// if x {
    ///     false
    /// } else {
    ///     true
    /// }
    /// # ;
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// # let x = true;
    /// !x
    /// # ;
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub NEEDLESS_BOOL,
    complexity,
    "if-statements with plain booleans in the then- and else-clause, e.g., `if p { true } else { false }`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for expressions of the form `if c { x = true } else { x = false }`
    /// (or vice versa) and suggest assigning the variable directly from the
    /// condition.
    ///
    /// ### Why is this bad?
    /// Redundant code.
    ///
    /// ### Example
    /// ```rust,ignore
    /// # fn must_keep(x: i32, y: i32) -> bool { x == y }
    /// # let x = 32; let y = 10;
    /// # let mut skip: bool;
    /// if must_keep(x, y) {
    ///     skip = false;
    /// } else {
    ///     skip = true;
    /// }
    /// ```
    /// Use instead:
    /// ```rust,ignore
    /// # fn must_keep(x: i32, y: i32) -> bool { x == y }
    /// # let x = 32; let y = 10;
    /// # let mut skip: bool;
    /// skip = !must_keep(x, y);
    /// ```
    #[clippy::version = "1.71.0"]
    pub NEEDLESS_BOOL_ASSIGN,
    complexity,
    "setting the same boolean variable in both branches of an if-statement"
}
declare_lint_pass!(NeedlessBool => [NEEDLESS_BOOL, NEEDLESS_BOOL_ASSIGN]);

fn condition_needs_parentheses(e: &Expr<'_>) -> bool {
    let mut inner = e;
    while let ExprKind::Binary(_, i, _)
    | ExprKind::Call(i, _)
    | ExprKind::Cast(i, _)
    | ExprKind::Type(i, _)
    | ExprKind::Index(i, _, _) = inner.kind
    {
        if is_block_like(i) {
            return true;
        }
        inner = i;
    }
    false
}

impl<'tcx> LateLintPass<'tcx> for NeedlessBool {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, e: &'tcx Expr<'_>) {
        use self::Expression::{Bool, RetBool};
        if !e.span.from_expansion()
            && let Some(higher::If {
                cond,
                then,
                r#else: Some(else_expr),
            }) = higher::If::hir(e)
            && !span_contains_comment(cx.tcx.sess.source_map(), e.span)
        {
            let reduce = |ret, not| {
                let mut applicability = Applicability::MachineApplicable;
                let snip = Sugg::hir_with_applicability(cx, cond, "<predicate>", &mut applicability);
                let mut snip = if not { !snip } else { snip };

                if ret {
                    snip = snip.make_return();
                }

                if is_else_clause(cx.tcx, e) {
                    snip = snip.blockify();
                }

                if (condition_needs_parentheses(cond) && is_parent_stmt(cx, e.hir_id))
                    || is_receiver_of_method_call(cx, e)
                    || is_as_argument(cx, e)
                {
                    snip = snip.maybe_paren();
                }

                span_lint_and_sugg(
                    cx,
                    NEEDLESS_BOOL,
                    e.span,
                    "this if-then-else expression returns a bool literal",
                    "you can reduce it to",
                    snip.to_string(),
                    applicability,
                );
            };
            if let Some(a) = fetch_bool_block(then)
                && let Some(b) = fetch_bool_block(else_expr)
            {
                match (a, b) {
                    (RetBool(true), RetBool(true)) | (Bool(true), Bool(true)) => {
                        span_lint(
                            cx,
                            NEEDLESS_BOOL,
                            e.span,
                            "this if-then-else expression will always return true",
                        );
                    },
                    (RetBool(false), RetBool(false)) | (Bool(false), Bool(false)) => {
                        span_lint(
                            cx,
                            NEEDLESS_BOOL,
                            e.span,
                            "this if-then-else expression will always return false",
                        );
                    },
                    (RetBool(true), RetBool(false)) => reduce(true, false),
                    (Bool(true), Bool(false)) => reduce(false, false),
                    (RetBool(false), RetBool(true)) => reduce(true, true),
                    (Bool(false), Bool(true)) => reduce(false, true),
                    _ => (),
                }
            }
            if let Some((lhs_a, a)) = fetch_assign(then)
                && let Some((lhs_b, b)) = fetch_assign(else_expr)
                && SpanlessEq::new(cx).eq_expr(lhs_a, lhs_b)
            {
                let mut applicability = Applicability::MachineApplicable;
                let cond = Sugg::hir_with_applicability(cx, cond, "..", &mut applicability);
                let lhs = snippet_with_applicability(cx, lhs_a.span, "..", &mut applicability);
                let mut sugg = if a == b {
                    format!("{cond}; {lhs} = {a:?};")
                } else {
                    format!("{lhs} = {};", if a { cond } else { !cond })
                };

                if is_else_clause(cx.tcx, e) {
                    sugg = format!("{{ {sugg} }}");
                }

                span_lint_and_sugg(
                    cx,
                    NEEDLESS_BOOL_ASSIGN,
                    e.span,
                    "this if-then-else expression assigns a bool literal",
                    "you can reduce it to",
                    sugg,
                    applicability,
                );
            }
        }
    }
}

enum Expression {
    Bool(bool),
    RetBool(bool),
}

fn fetch_bool_block(expr: &Expr<'_>) -> Option<Expression> {
    match peel_blocks_with_stmt(expr).kind {
        ExprKind::Ret(Some(ret)) => Some(Expression::RetBool(fetch_bool_expr(ret)?)),
        _ => Some(Expression::Bool(fetch_bool_expr(expr)?)),
    }
}

fn fetch_bool_expr(expr: &Expr<'_>) -> Option<bool> {
    if let ExprKind::Lit(lit_ptr) = peel_blocks(expr).kind
        && let LitKind::Bool(value) = lit_ptr.node
    {
        return Some(value);
    }
    None
}

fn fetch_assign<'tcx>(expr: &'tcx Expr<'tcx>) -> Option<(&'tcx Expr<'tcx>, bool)> {
    if let ExprKind::Assign(lhs, rhs, _) = peel_blocks_with_stmt(expr).kind {
        fetch_bool_expr(rhs).map(|b| (lhs, b))
    } else {
        None
    }
}

fn is_as_argument(cx: &LateContext<'_>, e: &Expr<'_>) -> bool {
    matches!(get_parent_expr(cx, e).map(|e| e.kind), Some(ExprKind::Cast(_, _)))
}
