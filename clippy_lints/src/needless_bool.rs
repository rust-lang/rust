use clippy_utils::diagnostics::{span_lint, span_lint_and_sugg, span_lint_and_then};
use clippy_utils::source::snippet_with_context;
use clippy_utils::sugg::Sugg;
use clippy_utils::{
    SpanlessEq, get_parent_expr, higher, is_block_like, is_else_clause, is_parent_stmt, is_receiver_of_method_call,
    peel_blocks, peel_blocks_with_stmt, span_contains_comment,
};
use rustc_ast::ast::LitKind;
use rustc_errors::Applicability;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::DefId;
use rustc_hir::{Block, Expr, ExprKind, StmtKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::declare_lint_pass;
use rustc_span::SyntaxContext;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for expressions of the form `if c { true } else {
    /// false }` (or vice versa) and suggests using the condition directly.
    ///
    /// This also covers the early-return guard form, where a condition returns a
    /// tuple-like based on it.
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
    ///
    /// Or, for the early-return guard form:
    /// ```
    /// # fn f(c: bool) -> Result<bool, ()> {
    /// if c {
    ///     return Ok(true);
    /// }
    /// Ok(false)
    /// # }
    /// ```
    ///
    /// Use instead:
    /// ```
    /// # fn f(c: bool) -> Result<bool, ()> {
    /// Ok(c)
    /// # }
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
            && !span_contains_comment(cx, e.span)
        {
            let reduce = |ret, not| {
                span_lint_and_then(
                    cx,
                    NEEDLESS_BOOL,
                    e.span,
                    "this if-then-else expression returns a bool literal",
                    |diag| {
                        let mut applicability = Applicability::MachineApplicable;
                        let snip = Sugg::hir_with_context(cx, cond, e.span.ctxt(), "<predicate>", &mut applicability);
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
                            || is_operand_of_binary_or_unary(cx, e)
                        {
                            snip = snip.maybe_paren();
                        }

                        diag.span_suggestion(e.span, "you can reduce it to", snip.to_string(), applicability);
                    },
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
                && SpanlessEq::new(cx).eq_expr(SyntaxContext::root(), lhs_a, lhs_b)
            {
                let mut applicability = Applicability::MachineApplicable;
                let cond = Sugg::hir_with_context(cx, cond, e.span.ctxt(), "..", &mut applicability);
                let (lhs, _) = snippet_with_context(cx, lhs_a.span, e.span.ctxt(), "..", &mut applicability);
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

    fn check_block(&mut self, cx: &LateContext<'tcx>, block: &'tcx Block<'tcx>) {
        // Detect the early-return form:
        //     if c {
        //         return Ok(true);
        //     }
        //     Ok(false)
        // and reduce it to `Ok(c)`. The optional `Ok(..)` wrapper can be any tuple-like
        // constructor (or absent), as long as the guard and the trailing expression use the
        // same one. Constructors are pure, so folding the condition in keeps the behavior.
        if let Some(tail) = block.expr
            && !tail.span.from_expansion()
            && let [.., last_stmt] = block.stmts
            && let StmtKind::Semi(if_expr) | StmtKind::Expr(if_expr) = last_stmt.kind
            && !if_expr.span.from_expansion()
            && let Some(higher::If {
                cond,
                then,
                r#else: None,
            }) = higher::If::hir(if_expr)
            && let ExprKind::Ret(Some(ret)) = peel_blocks_with_stmt(then).kind
            && let Some((then_func, then_val)) = fetch_bool_and_wrapper(ret)
            && let Some((tail_func, tail_val)) = fetch_bool_and_wrapper(tail)
            // `if c { return Ok(true) } Ok(true)` is always the same value; the condition might
            // have side effects, so don't touch it. Comparing the values here is cheap, so do it
            // before the constructor resolution in `wrapper_ctors_match`, which needs `qpath_res`.
            && then_val != tail_val
            && wrapper_ctors_match(cx, then_func, tail_func)
        {
            let span = if_expr.span.to(tail.span);
            if span_contains_comment(cx, span) {
                return;
            }

            let mut applicability = Applicability::MachineApplicable;
            let mut snip = Sugg::hir_with_context(cx, cond, span.ctxt(), "..", &mut applicability);
            // `then_val` is the value returned when the condition holds, so a `false` there means
            // the result is the negation of the condition.
            if !then_val {
                snip = !snip;
            }
            let sugg = match tail_func {
                Some(func) => {
                    let func_snip = snippet_with_context(cx, func.span, span.ctxt(), "..", &mut applicability).0;
                    format!("{func_snip}({snip})")
                },
                None => snip.to_string(),
            };

            span_lint_and_sugg(
                cx,
                NEEDLESS_BOOL,
                span,
                "this `if` guard returns a bool literal and is followed by another",
                "you can reduce it to",
                sugg,
                applicability,
            );
        }
    }
}

/// Returns the optional constructor call wrapping a bool literal (e.g. the `Ok` in `Ok(true)`)
/// along with the bool value. The constructor is returned as the unresolved callee expression;
/// resolving it to a `DefId` (via [`wrapper_ctors_match`]) needs `qpath_res`, which is
/// comparatively expensive, so callers should only do that once the cheap bool-value comparison
/// has passed. A bare bool literal yields `None`.
fn fetch_bool_and_wrapper<'tcx>(expr: &'tcx Expr<'tcx>) -> Option<(Option<&'tcx Expr<'tcx>>, bool)> {
    let expr = peel_blocks(expr);
    if let Some(value) = fetch_bool_expr(expr) {
        return Some((None, value));
    }
    if let ExprKind::Call(func, [arg]) = expr.kind
        && let Some(value) = fetch_bool_expr(arg)
    {
        return Some((Some(func), value));
    }
    None
}

/// Checks whether two optional wrapper-constructor call expressions (as returned by
/// [`fetch_bool_and_wrapper`]) are absent on both sides, or call the same tuple-like constructor.
fn wrapper_ctors_match(cx: &LateContext<'_>, a: Option<&Expr<'_>>, b: Option<&Expr<'_>>) -> bool {
    fn resolve_ctor(cx: &LateContext<'_>, func: &Expr<'_>) -> Option<DefId> {
        if let ExprKind::Path(qpath) = &func.kind
            && let Res::Def(DefKind::Ctor(..), did) = cx.qpath_res(qpath, func.hir_id)
        {
            return Some(did);
        }
        None
    }
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => resolve_ctor(cx, a).is_some_and(|a| Some(a) == resolve_ctor(cx, b)),
        _ => false,
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

fn is_operand_of_binary_or_unary(cx: &LateContext<'_>, e: &Expr<'_>) -> bool {
    matches!(
        get_parent_expr(cx, e).map(|e| e.kind),
        Some(ExprKind::Binary(..) | ExprKind::Unary(..))
    )
}
