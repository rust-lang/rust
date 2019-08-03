//! Checks for needless boolean results of if-else expressions
//!
//! This lint is **warn** by default

use crate::utils::sugg::Sugg;
use crate::utils::{higher, in_macro_or_desugar, span_lint, span_lint_and_sugg};
use rustc::hir::*;
use rustc::lint::{LateContext, LateLintPass, LintArray, LintPass};
use rustc::{declare_lint_pass, declare_tool_lint};
use rustc_errors::Applicability;
use syntax::ast::LitKind;
use syntax::source_map::Spanned;

declare_clippy_lint! {
    /// **What it does:** Checks for expressions of the form `if c { true } else {
    /// false }`
    /// (or vice versa) and suggest using the condition directly.
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
    /// if x == true {} // could be `if x { }`
    /// ```
    pub BOOL_COMPARISON,
    complexity,
    "comparing a variable to a boolean, e.g., `if x == true` or `if x != true`"
}

declare_lint_pass!(NeedlessBool => [NEEDLESS_BOOL]);

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for NeedlessBool {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, e: &'tcx Expr) {
        use self::Expression::*;
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
            if let ExprKind::Block(ref then_block, _) = then_block.node {
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
                panic!("IfExpr 'then' node is not an ExprKind::Block");
            }
        }
    }
}

fn parent_node_is_if_expr<'a, 'b>(expr: &Expr, cx: &LateContext<'a, 'b>) -> bool {
    let parent_id = cx.tcx.hir().get_parent_node(expr.hir_id);
    let parent_node = cx.tcx.hir().get(parent_id);

    if let rustc::hir::Node::Expr(e) = parent_node {
        if higher::if_block(&e).is_some() {
            return true;
        }
    }

    false
}

declare_lint_pass!(BoolComparison => [BOOL_COMPARISON]);

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for BoolComparison {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, e: &'tcx Expr) {
        if in_macro_or_desugar(e.span) {
            return;
        }

        if let ExprKind::Binary(Spanned { node, .. }, ..) = e.node {
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

fn check_comparison<'a, 'tcx>(
    cx: &LateContext<'a, 'tcx>,
    e: &'tcx Expr,
    left_true: Option<(impl FnOnce(Sugg<'a>) -> Sugg<'a>, &str)>,
    left_false: Option<(impl FnOnce(Sugg<'a>) -> Sugg<'a>, &str)>,
    right_true: Option<(impl FnOnce(Sugg<'a>) -> Sugg<'a>, &str)>,
    right_false: Option<(impl FnOnce(Sugg<'a>) -> Sugg<'a>, &str)>,
    no_literal: Option<(impl FnOnce(Sugg<'a>, Sugg<'a>) -> Sugg<'a>, &str)>,
) {
    use self::Expression::*;

    if let ExprKind::Binary(_, ref left_side, ref right_side) = e.node {
        let (l_ty, r_ty) = (cx.tables.expr_ty(left_side), cx.tables.expr_ty(right_side));
        if l_ty.is_bool() && r_ty.is_bool() {
            let mut applicability = Applicability::MachineApplicable;
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
    cx: &LateContext<'a, 'tcx>,
    e: &'tcx Expr,
    expr: &Expr,
    mut applicability: Applicability,
    message: &str,
    conv_hint: impl FnOnce(Sugg<'a>) -> Sugg<'a>,
) {
    let hint = Sugg::hir_with_applicability(cx, expr, "..", &mut applicability);
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

fn fetch_bool_block(block: &Block) -> Expression {
    match (&*block.stmts, block.expr.as_ref()) {
        (&[], Some(e)) => fetch_bool_expr(&**e),
        (&[ref e], None) => {
            if let StmtKind::Semi(ref e) = e.node {
                if let ExprKind::Ret(_) = e.node {
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

fn fetch_bool_expr(expr: &Expr) -> Expression {
    match expr.node {
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
