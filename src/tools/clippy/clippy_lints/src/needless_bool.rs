use clippy_utils::diagnostics::{span_lint, span_lint_and_sugg};
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::sugg::Sugg;
use clippy_utils::{
    SpanlessEq, get_parent_expr, higher, is_block_like, is_else_clause, is_expn_of, is_parent_stmt,
    is_receiver_of_method_call, peel_blocks, peel_blocks_with_stmt, span_extract_comment,
};
use rustc_ast::ast::LitKind;
use rustc_errors::Applicability;
use rustc_hir::{BinOpKind, Expr, ExprKind, UnOp};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::declare_lint_pass;
use rustc_span::Span;
use rustc_span::source_map::Spanned;

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
    /// Checks for expressions of the form `x == true`,
    /// `x != true` and order comparisons such as `x < true` (or vice versa) and
    /// suggest using the variable directly.
    ///
    /// ### Why is this bad?
    /// Unnecessary code.
    ///
    /// ### Example
    /// ```rust,ignore
    /// if x == true {}
    /// if y == false {}
    /// ```
    /// use `x` directly:
    /// ```rust,ignore
    /// if x {}
    /// if !y {}
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub BOOL_COMPARISON,
    complexity,
    "comparing a variable to a boolean, e.g., `if x == true` or `if x != true`"
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
        if e.span.from_expansion() || !span_extract_comment(cx.tcx.sess.source_map(), e.span).is_empty() {
            return;
        }
        if let Some(higher::If {
            cond,
            then,
            r#else: Some(r#else),
        }) = higher::If::hir(e)
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
            if let Some((a, b)) = fetch_bool_block(then).and_then(|a| Some((a, fetch_bool_block(r#else)?))) {
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
                && let Some((lhs_b, b)) = fetch_assign(r#else)
                && SpanlessEq::new(cx).eq_expr(lhs_a, lhs_b)
            {
                let mut applicability = Applicability::MachineApplicable;
                let cond = Sugg::hir_with_applicability(cx, cond, "..", &mut applicability);
                let lhs = snippet_with_applicability(cx, lhs_a.span, "..", &mut applicability);
                let sugg = if a == b {
                    format!("{cond}; {lhs} = {a:?};")
                } else {
                    format!("{lhs} = {};", if a { cond } else { !cond })
                };
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

declare_lint_pass!(BoolComparison => [BOOL_COMPARISON]);

impl<'tcx> LateLintPass<'tcx> for BoolComparison {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, e: &'tcx Expr<'_>) {
        if e.span.from_expansion() {
            return;
        }

        if let ExprKind::Binary(Spanned { node, .. }, ..) = e.kind {
            let ignore_case = None::<(fn(_) -> _, &str)>;
            let ignore_no_literal = None::<(fn(_, _) -> _, &str)>;
            match node {
                BinOpKind::Eq => {
                    let true_case = Some((|h| h, "equality checks against true are unnecessary"));
                    let false_case = Some((
                        |h: Sugg<'tcx>| !h,
                        "equality checks against false can be replaced by a negation",
                    ));
                    check_comparison(cx, e, true_case, false_case, true_case, false_case, ignore_no_literal);
                },
                BinOpKind::Ne => {
                    let true_case = Some((
                        |h: Sugg<'tcx>| !h,
                        "inequality checks against true can be replaced by a negation",
                    ));
                    let false_case = Some((|h| h, "inequality checks against false are unnecessary"));
                    check_comparison(cx, e, true_case, false_case, true_case, false_case, ignore_no_literal);
                },
                BinOpKind::Lt => check_comparison(
                    cx,
                    e,
                    ignore_case,
                    Some((|h| h, "greater than checks against false are unnecessary")),
                    Some((
                        |h: Sugg<'tcx>| !h,
                        "less than comparison against true can be replaced by a negation",
                    )),
                    ignore_case,
                    Some((
                        |l: Sugg<'tcx>, r: Sugg<'tcx>| (!l).bit_and(&r),
                        "order comparisons between booleans can be simplified",
                    )),
                ),
                BinOpKind::Gt => check_comparison(
                    cx,
                    e,
                    Some((
                        |h: Sugg<'tcx>| !h,
                        "less than comparison against true can be replaced by a negation",
                    )),
                    ignore_case,
                    ignore_case,
                    Some((|h| h, "greater than checks against false are unnecessary")),
                    Some((
                        |l: Sugg<'tcx>, r: Sugg<'tcx>| l.bit_and(&(!r)),
                        "order comparisons between booleans can be simplified",
                    )),
                ),
                _ => (),
            }
        }
    }
}

struct ExpressionInfoWithSpan {
    one_side_is_unary_not: bool,
    left_span: Span,
    right_span: Span,
}

fn is_unary_not(e: &Expr<'_>) -> (bool, Span) {
    if let ExprKind::Unary(UnOp::Not, operand) = e.kind {
        return (true, operand.span);
    }
    (false, e.span)
}

fn one_side_is_unary_not<'tcx>(left_side: &'tcx Expr<'_>, right_side: &'tcx Expr<'_>) -> ExpressionInfoWithSpan {
    let left = is_unary_not(left_side);
    let right = is_unary_not(right_side);

    ExpressionInfoWithSpan {
        one_side_is_unary_not: left.0 != right.0,
        left_span: left.1,
        right_span: right.1,
    }
}

fn check_comparison<'a, 'tcx>(
    cx: &LateContext<'tcx>,
    e: &'tcx Expr<'_>,
    left_true: Option<(impl FnOnce(Sugg<'a>) -> Sugg<'a>, &'static str)>,
    left_false: Option<(impl FnOnce(Sugg<'a>) -> Sugg<'a>, &'static str)>,
    right_true: Option<(impl FnOnce(Sugg<'a>) -> Sugg<'a>, &'static str)>,
    right_false: Option<(impl FnOnce(Sugg<'a>) -> Sugg<'a>, &'static str)>,
    no_literal: Option<(impl FnOnce(Sugg<'a>, Sugg<'a>) -> Sugg<'a>, &'static str)>,
) {
    if let ExprKind::Binary(op, left_side, right_side) = e.kind {
        let (l_ty, r_ty) = (
            cx.typeck_results().expr_ty(left_side),
            cx.typeck_results().expr_ty(right_side),
        );
        if is_expn_of(left_side.span, "cfg").is_some() || is_expn_of(right_side.span, "cfg").is_some() {
            return;
        }
        if l_ty.is_bool() && r_ty.is_bool() {
            let mut applicability = Applicability::MachineApplicable;
            // Eliminate parentheses in `e` by using the lo pos of lhs and hi pos of rhs,
            // calling `source_callsite` make sure macros are handled correctly, see issue #9907
            let binop_span = left_side
                .span
                .source_callsite()
                .with_hi(right_side.span.source_callsite().hi());

            if op.node == BinOpKind::Eq {
                let expression_info = one_side_is_unary_not(left_side, right_side);
                if expression_info.one_side_is_unary_not {
                    span_lint_and_sugg(
                        cx,
                        BOOL_COMPARISON,
                        binop_span,
                        "this comparison might be written more concisely",
                        "try simplifying it as shown",
                        format!(
                            "{} != {}",
                            snippet_with_applicability(
                                cx,
                                expression_info.left_span.source_callsite(),
                                "..",
                                &mut applicability
                            ),
                            snippet_with_applicability(
                                cx,
                                expression_info.right_span.source_callsite(),
                                "..",
                                &mut applicability
                            )
                        ),
                        applicability,
                    );
                }
            }

            match (fetch_bool_expr(left_side), fetch_bool_expr(right_side)) {
                (Some(true), None) => left_true.map_or((), |(h, m)| {
                    suggest_bool_comparison(cx, binop_span, right_side, applicability, m, h);
                }),
                (None, Some(true)) => right_true.map_or((), |(h, m)| {
                    suggest_bool_comparison(cx, binop_span, left_side, applicability, m, h);
                }),
                (Some(false), None) => left_false.map_or((), |(h, m)| {
                    suggest_bool_comparison(cx, binop_span, right_side, applicability, m, h);
                }),
                (None, Some(false)) => right_false.map_or((), |(h, m)| {
                    suggest_bool_comparison(cx, binop_span, left_side, applicability, m, h);
                }),
                (None, None) => no_literal.map_or((), |(h, m)| {
                    let left_side = Sugg::hir_with_applicability(cx, left_side, "..", &mut applicability);
                    let right_side = Sugg::hir_with_applicability(cx, right_side, "..", &mut applicability);
                    span_lint_and_sugg(
                        cx,
                        BOOL_COMPARISON,
                        binop_span,
                        m,
                        "try simplifying it as shown",
                        h(left_side, right_side).into_string(),
                        applicability,
                    );
                }),
                _ => (),
            }
        }
    }
}

fn suggest_bool_comparison<'a, 'tcx>(
    cx: &LateContext<'tcx>,
    span: Span,
    expr: &Expr<'_>,
    mut app: Applicability,
    message: &'static str,
    conv_hint: impl FnOnce(Sugg<'a>) -> Sugg<'a>,
) {
    let hint = Sugg::hir_with_context(cx, expr, span.ctxt(), "..", &mut app);
    span_lint_and_sugg(
        cx,
        BOOL_COMPARISON,
        span,
        message,
        "try simplifying it as shown",
        conv_hint(hint).into_string(),
        app,
    );
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
