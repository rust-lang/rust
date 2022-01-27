use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::snippet_opt;
use clippy_utils::ty::implements_trait;
use clippy_utils::{binop_traits, sugg};
use clippy_utils::{eq_expr_value, trait_ref_of_method};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_hir::intravisit::{walk_expr, Visitor};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `a = a op b` or `a = b commutative_op a`
    /// patterns.
    ///
    /// ### Why is this bad?
    /// These can be written as the shorter `a op= b`.
    ///
    /// ### Known problems
    /// While forbidden by the spec, `OpAssign` traits may have
    /// implementations that differ from the regular `Op` impl.
    ///
    /// ### Example
    /// ```rust
    /// let mut a = 5;
    /// let b = 0;
    /// // ...
    /// // Bad
    /// a = a + b;
    ///
    /// // Good
    /// a += b;
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub ASSIGN_OP_PATTERN,
    style,
    "assigning the result of an operation on a variable to that same variable"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `a op= a op b` or `a op= b op a` patterns.
    ///
    /// ### Why is this bad?
    /// Most likely these are bugs where one meant to write `a
    /// op= b`.
    ///
    /// ### Known problems
    /// Clippy cannot know for sure if `a op= a op b` should have
    /// been `a = a op a op b` or `a = a op b`/`a op= b`. Therefore, it suggests both.
    /// If `a op= a op b` is really the correct behaviour it should be
    /// written as `a = a op a op b` as it's less confusing.
    ///
    /// ### Example
    /// ```rust
    /// let mut a = 5;
    /// let b = 2;
    /// // ...
    /// a += a + b;
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub MISREFACTORED_ASSIGN_OP,
    suspicious,
    "having a variable on both sides of an assign op"
}

declare_lint_pass!(AssignOps => [ASSIGN_OP_PATTERN, MISREFACTORED_ASSIGN_OP]);

impl<'tcx> LateLintPass<'tcx> for AssignOps {
    #[allow(clippy::too_many_lines)]
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx hir::Expr<'_>) {
        match &expr.kind {
            hir::ExprKind::AssignOp(op, lhs, rhs) => {
                if let hir::ExprKind::Binary(binop, l, r) = &rhs.kind {
                    if op.node != binop.node {
                        return;
                    }
                    // lhs op= l op r
                    if eq_expr_value(cx, lhs, l) {
                        lint_misrefactored_assign_op(cx, expr, *op, rhs, lhs, r);
                    }
                    // lhs op= l commutative_op r
                    if is_commutative(op.node) && eq_expr_value(cx, lhs, r) {
                        lint_misrefactored_assign_op(cx, expr, *op, rhs, lhs, l);
                    }
                }
            },
            hir::ExprKind::Assign(assignee, e, _) => {
                if let hir::ExprKind::Binary(op, l, r) = &e.kind {
                    let lint = |assignee: &hir::Expr<'_>, rhs: &hir::Expr<'_>| {
                        let ty = cx.typeck_results().expr_ty(assignee);
                        let rty = cx.typeck_results().expr_ty(rhs);
                        if_chain! {
                            if let Some((_, lang_item)) = binop_traits(op.node);
                            if let Ok(trait_id) = cx.tcx.lang_items().require(lang_item);
                            let parent_fn = cx.tcx.hir().get_parent_item(e.hir_id);
                            if trait_ref_of_method(cx, parent_fn)
                                .map_or(true, |t| t.path.res.def_id() != trait_id);
                            if implements_trait(cx, ty, trait_id, &[rty.into()]);
                            then {
                                span_lint_and_then(
                                    cx,
                                    ASSIGN_OP_PATTERN,
                                    expr.span,
                                    "manual implementation of an assign operation",
                                    |diag| {
                                        if let (Some(snip_a), Some(snip_r)) =
                                            (snippet_opt(cx, assignee.span), snippet_opt(cx, rhs.span))
                                        {
                                            diag.span_suggestion(
                                                expr.span,
                                                "replace it with",
                                                format!("{} {}= {}", snip_a, op.node.as_str(), snip_r),
                                                Applicability::MachineApplicable,
                                            );
                                        }
                                    },
                                );
                            }
                        }
                    };

                    let mut visitor = ExprVisitor {
                        assignee,
                        counter: 0,
                        cx,
                    };

                    walk_expr(&mut visitor, e);

                    if visitor.counter == 1 {
                        // a = a op b
                        if eq_expr_value(cx, assignee, l) {
                            lint(assignee, r);
                        }
                        // a = b commutative_op a
                        // Limited to primitive type as these ops are know to be commutative
                        if eq_expr_value(cx, assignee, r) && cx.typeck_results().expr_ty(assignee).is_primitive_ty() {
                            match op.node {
                                hir::BinOpKind::Add
                                | hir::BinOpKind::Mul
                                | hir::BinOpKind::And
                                | hir::BinOpKind::Or
                                | hir::BinOpKind::BitXor
                                | hir::BinOpKind::BitAnd
                                | hir::BinOpKind::BitOr => {
                                    lint(assignee, l);
                                },
                                _ => {},
                            }
                        }
                    }
                }
            },
            _ => {},
        }
    }
}

fn lint_misrefactored_assign_op(
    cx: &LateContext<'_>,
    expr: &hir::Expr<'_>,
    op: hir::BinOp,
    rhs: &hir::Expr<'_>,
    assignee: &hir::Expr<'_>,
    rhs_other: &hir::Expr<'_>,
) {
    span_lint_and_then(
        cx,
        MISREFACTORED_ASSIGN_OP,
        expr.span,
        "variable appears on both sides of an assignment operation",
        |diag| {
            if let (Some(snip_a), Some(snip_r)) = (snippet_opt(cx, assignee.span), snippet_opt(cx, rhs_other.span)) {
                let a = &sugg::Sugg::hir(cx, assignee, "..");
                let r = &sugg::Sugg::hir(cx, rhs, "..");
                let long = format!("{} = {}", snip_a, sugg::make_binop(op.node.into(), a, r));
                diag.span_suggestion(
                    expr.span,
                    &format!(
                        "did you mean `{} = {} {} {}` or `{}`? Consider replacing it with",
                        snip_a,
                        snip_a,
                        op.node.as_str(),
                        snip_r,
                        long
                    ),
                    format!("{} {}= {}", snip_a, op.node.as_str(), snip_r),
                    Applicability::MaybeIncorrect,
                );
                diag.span_suggestion(
                    expr.span,
                    "or",
                    long,
                    Applicability::MaybeIncorrect, // snippet
                );
            }
        },
    );
}

#[must_use]
fn is_commutative(op: hir::BinOpKind) -> bool {
    use rustc_hir::BinOpKind::{
        Add, And, BitAnd, BitOr, BitXor, Div, Eq, Ge, Gt, Le, Lt, Mul, Ne, Or, Rem, Shl, Shr, Sub,
    };
    match op {
        Add | Mul | And | Or | BitXor | BitAnd | BitOr | Eq | Ne => true,
        Sub | Div | Rem | Shl | Shr | Lt | Le | Ge | Gt => false,
    }
}

struct ExprVisitor<'a, 'tcx> {
    assignee: &'a hir::Expr<'a>,
    counter: u8,
    cx: &'a LateContext<'tcx>,
}

impl<'a, 'tcx> Visitor<'tcx> for ExprVisitor<'a, 'tcx> {
    fn visit_expr(&mut self, expr: &'tcx hir::Expr<'_>) {
        if eq_expr_value(self.cx, self.assignee, expr) {
            self.counter += 1;
        }

        walk_expr(self, expr);
    }
}
