//! Checks for needless boolean results of if-else expressions
//!
//! This lint is **warn** by default

use crate::utils::sugg::Sugg;
use crate::utils::{
    higher, is_expn_of, parent_node_is_if_expr, snippet_with_applicability, span_lint, span_lint_and_sugg,
};
use rustc_ast::ast::LitKind;
use rustc_errors::Applicability;
use rustc_hir::{BinOpKind, Block, Expr, ExprKind, StmtKind, UnOp};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::source_map::Spanned;
use rustc_span::Span;

declare_clippy_lint! {
    /// **What it does:** Checks for expressions of the form `if c { true } else {
    /// false }` (or vice versa) and suggests using the condition directly.
    ///
    /// **Why is this bad?** Redundant code.
    ///
    /// **Known problems:** Maybe false positives: Sometimes, the two branches are
    /// painstakingly documented (which we, of course, do not detect), so they *may*
    /// have some value. Even then, the documentation can be rewritten to match the
    /// shorter code.
    ///
    /// **Example:**
    /// ```rust,ignore
    /// if x {
    ///     false
    /// } else {
    ///     true
    /// }
    /// ```
    /// Could be written as
    /// ```rust,ignore
    /// !x
    /// ```
    pub NEEDLESS_BOOL,
    complexity,
    "if-statements with plain booleans in the then- and else-clause, e.g., `if p { true } else { false }`"
}

declare_clippy_lint! {
    /// **What it does:** Checks for expressions of the form `x == true`,
    /// `x != true` and order comparisons such as `x < true` (or vice versa) and
    /// suggest using the variable directly.
    ///
    /// **Why is this bad?** Unnecessary code.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust,ignore
    /// if x == true {}
    /// if y == false {}
    /// ```
    /// use `x` directly:
    /// ```rust,ignore
    /// if x {}
    /// if !y {}
    /// ```
    pub BOOL_COMPARISON,
    complexity,
    "comparing a variable to a boolean, e.g., `if x == true` or `if x != true`"
}

declare_lint_pass!(NeedlessBool => [NEEDLESS_BOOL]);

impl<'tcx> LateLintPass<'tcx> for NeedlessBool {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, e: &'tcx Expr<'_>) {
        use self::Expression::{Bool, RetBool};
        if let Some((ref pred, ref then_block, Some(ref else_expr))) = higher::if_block(&e) {
            let reduce = |ret, not| {
                let mut applicability = Applicability::MachineApplicable;
                let snip = Sugg::hir_with_applicability(cx, pred, "<predicate>", &mut applicability);
                let mut snip = if not { !snip } else { snip };

                if ret {
                    snip = snip.make_return();
                }

                if parent_node_is_if_expr(&e, &cx) {
                    snip = snip.blockify()
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
            if let ExprKind::Block(ref then_block, _) = then_block.kind {
                match (fetch_bool_block(then_block), fetch_bool_expr(else_expr)) {
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
            } else {
                panic!("IfExpr `then` node is not an `ExprKind::Block`");
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
                        |h: Sugg<'_>| !h,
                        "equality checks against false can be replaced by a negation",
                    ));
                    check_comparison(cx, e, true_case, false_case, true_case, false_case, ignore_no_literal)
                },
                BinOpKind::Ne => {
                    let true_case = Some((
                        |h: Sugg<'_>| !h,
                        "inequality checks against true can be replaced by a negation",
                    ));
                    let false_case = Some((|h| h, "inequality checks against false are unnecessary"));
                    check_comparison(cx, e, true_case, false_case, true_case, false_case, ignore_no_literal)
                },
                BinOpKind::Lt => check_comparison(
                    cx,
                    e,
                    ignore_case,
                    Some((|h| h, "greater than checks against false are unnecessary")),
                    Some((
                        |h: Sugg<'_>| !h,
                        "less than comparison against true can be replaced by a negation",
                    )),
                    ignore_case,
                    Some((
                        |l: Sugg<'_>, r: Sugg<'_>| (!l).bit_and(&r),
                        "order comparisons between booleans can be simplified",
                    )),
                ),
                BinOpKind::Gt => check_comparison(
                    cx,
                    e,
                    Some((
                        |h: Sugg<'_>| !h,
                        "less than comparison against true can be replaced by a negation",
                    )),
                    ignore_case,
                    ignore_case,
                    Some((|h| h, "greater than checks against false are unnecessary")),
                    Some((
                        |l: Sugg<'_>, r: Sugg<'_>| l.bit_and(&(!r)),
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
    if let ExprKind::Unary(UnOp::UnNot, operand) = e.kind {
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
    left_true: Option<(impl FnOnce(Sugg<'a>) -> Sugg<'a>, &str)>,
    left_false: Option<(impl FnOnce(Sugg<'a>) -> Sugg<'a>, &str)>,
    right_true: Option<(impl FnOnce(Sugg<'a>) -> Sugg<'a>, &str)>,
    right_false: Option<(impl FnOnce(Sugg<'a>) -> Sugg<'a>, &str)>,
    no_literal: Option<(impl FnOnce(Sugg<'a>, Sugg<'a>) -> Sugg<'a>, &str)>,
) {
    use self::Expression::{Bool, Other};

    if let ExprKind::Binary(op, ref left_side, ref right_side) = e.kind {
        let (l_ty, r_ty) = (
            cx.typeck_results().expr_ty(left_side),
            cx.typeck_results().expr_ty(right_side),
        );
        if is_expn_of(left_side.span, "cfg").is_some() || is_expn_of(right_side.span, "cfg").is_some() {
            return;
        }
        if l_ty.is_bool() && r_ty.is_bool() {
            let mut applicability = Applicability::MachineApplicable;

            if let BinOpKind::Eq = op.node {
                let expression_info = one_side_is_unary_not(&left_side, &right_side);
                if expression_info.one_side_is_unary_not {
                    span_lint_and_sugg(
                        cx,
                        BOOL_COMPARISON,
                        e.span,
                        "this comparison might be written more concisely",
                        "try simplifying it as shown",
                        format!(
                            "{} != {}",
                            snippet_with_applicability(cx, expression_info.left_span, "..", &mut applicability),
                            snippet_with_applicability(cx, expression_info.right_span, "..", &mut applicability)
                        ),
                        applicability,
                    )
                }
            }

            match (fetch_bool_expr(left_side), fetch_bool_expr(right_side)) {
                (Bool(true), Other) => left_true.map_or((), |(h, m)| {
                    suggest_bool_comparison(cx, e, right_side, applicability, m, h)
                }),
                (Other, Bool(true)) => right_true.map_or((), |(h, m)| {
                    suggest_bool_comparison(cx, e, left_side, applicability, m, h)
                }),
                (Bool(false), Other) => left_false.map_or((), |(h, m)| {
                    suggest_bool_comparison(cx, e, right_side, applicability, m, h)
                }),
                (Other, Bool(false)) => right_false.map_or((), |(h, m)| {
                    suggest_bool_comparison(cx, e, left_side, applicability, m, h)
                }),
                (Other, Other) => no_literal.map_or((), |(h, m)| {
                    let left_side = Sugg::hir_with_applicability(cx, left_side, "..", &mut applicability);
                    let right_side = Sugg::hir_with_applicability(cx, right_side, "..", &mut applicability);
                    span_lint_and_sugg(
                        cx,
                        BOOL_COMPARISON,
                        e.span,
                        m,
                        "try simplifying it as shown",
                        h(left_side, right_side).to_string(),
                        applicability,
                    )
                }),
                _ => (),
            }
        }
    }
}

fn suggest_bool_comparison<'a, 'tcx>(
    cx: &LateContext<'tcx>,
    e: &'tcx Expr<'_>,
    expr: &Expr<'_>,
    mut applicability: Applicability,
    message: &str,
    conv_hint: impl FnOnce(Sugg<'a>) -> Sugg<'a>,
) {
    let hint = if expr.span.from_expansion() {
        if applicability != Applicability::Unspecified {
            applicability = Applicability::MaybeIncorrect;
        }
        Sugg::hir_with_macro_callsite(cx, expr, "..")
    } else {
        Sugg::hir_with_applicability(cx, expr, "..", &mut applicability)
    };
    span_lint_and_sugg(
        cx,
        BOOL_COMPARISON,
        e.span,
        message,
        "try simplifying it as shown",
        conv_hint(hint).to_string(),
        applicability,
    );
}

enum Expression {
    Bool(bool),
    RetBool(bool),
    Other,
}

fn fetch_bool_block(block: &Block<'_>) -> Expression {
    match (&*block.stmts, block.expr.as_ref()) {
        (&[], Some(e)) => fetch_bool_expr(&**e),
        (&[ref e], None) => {
            if let StmtKind::Semi(ref e) = e.kind {
                if let ExprKind::Ret(_) = e.kind {
                    fetch_bool_expr(&**e)
                } else {
                    Expression::Other
                }
            } else {
                Expression::Other
            }
        },
        _ => Expression::Other,
    }
}

fn fetch_bool_expr(expr: &Expr<'_>) -> Expression {
    match expr.kind {
        ExprKind::Block(ref block, _) => fetch_bool_block(block),
        ExprKind::Lit(ref lit_ptr) => {
            if let LitKind::Bool(value) = lit_ptr.node {
                Expression::Bool(value)
            } else {
                Expression::Other
            }
        },
        ExprKind::Ret(Some(ref expr)) => match fetch_bool_expr(expr) {
            Expression::Bool(value) => Expression::RetBool(value),
            _ => Expression::Other,
        },
        _ => Expression::Other,
    }
}
