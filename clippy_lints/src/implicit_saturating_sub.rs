use crate::utils::{higher, in_macro, match_qpath, span_lint_and_sugg, SpanlessEq};
use if_chain::if_chain;
use rustc_ast::ast::LitKind;
use rustc_errors::Applicability;
use rustc_hir::{BinOpKind, Expr, ExprKind, QPath, StmtKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// **What it does:** Checks for implicit saturating subtraction.
    ///
    /// **Why is this bad?** Simplicity and readability. Instead we can easily use an builtin function.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    ///
    /// ```rust
    /// let end: u32 = 10;
    /// let start: u32 = 5;
    ///
    /// let mut i: u32 = end - start;
    ///
    /// // Bad
    /// if i != 0 {
    ///     i -= 1;
    /// }
    ///
    /// // Good
    /// i = i.saturating_sub(1);
    /// ```
    pub IMPLICIT_SATURATING_SUB,
    pedantic,
    "Perform saturating subtraction instead of implicitly checking lower bound of data type"
}

declare_lint_pass!(ImplicitSaturatingSub => [IMPLICIT_SATURATING_SUB]);

impl<'tcx> LateLintPass<'tcx> for ImplicitSaturatingSub {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) {
        if in_macro(expr.span) {
            return;
        }
        if_chain! {
            if let Some((ref cond, ref then, None)) = higher::if_block(&expr);

            // Check if the conditional expression is a binary operation
            if let ExprKind::Binary(ref cond_op, ref cond_left, ref cond_right) = cond.kind;

            // Ensure that the binary operator is >, != and <
            if BinOpKind::Ne == cond_op.node || BinOpKind::Gt == cond_op.node || BinOpKind::Lt == cond_op.node;

            // Check if the true condition block has only one statement
            if let ExprKind::Block(ref block, _) = then.kind;
            if block.stmts.len() == 1 && block.expr.is_none();

            // Check if assign operation is done
            if let StmtKind::Semi(ref e) = block.stmts[0].kind;
            if let Some(target) = subtracts_one(cx, e);

            // Extracting out the variable name
            if let ExprKind::Path(QPath::Resolved(_, ref ares_path)) = target.kind;

            then {
                // Handle symmetric conditions in the if statement
                let (cond_var, cond_num_val) = if SpanlessEq::new(cx).eq_expr(cond_left, target) {
                    if BinOpKind::Gt == cond_op.node || BinOpKind::Ne == cond_op.node {
                        (cond_left, cond_right)
                    } else {
                        return;
                    }
                } else if SpanlessEq::new(cx).eq_expr(cond_right, target) {
                    if BinOpKind::Lt == cond_op.node || BinOpKind::Ne == cond_op.node {
                        (cond_right, cond_left)
                    } else {
                        return;
                    }
                } else {
                    return;
                };

                // Check if the variable in the condition statement is an integer
                if !cx.typeck_results().expr_ty(cond_var).is_integral() {
                    return;
                }

                // Get the variable name
                let var_name = ares_path.segments[0].ident.name.as_str();
                const INT_TYPES: [&str; 5] = ["i8", "i16", "i32", "i64", "i128"];

                match cond_num_val.kind {
                    ExprKind::Lit(ref cond_lit) => {
                        // Check if the constant is zero
                        if let LitKind::Int(0, _) = cond_lit.node {
                            if cx.typeck_results().expr_ty(cond_left).is_signed() {
                            } else {
                                print_lint_and_sugg(cx, &var_name, expr);
                            };
                        }
                    },
                    ExprKind::Path(ref cond_num_path) => {
                        if INT_TYPES.iter().any(|int_type| match_qpath(cond_num_path, &[int_type, "MIN"])) {
                            print_lint_and_sugg(cx, &var_name, expr);
                        };
                    },
                    ExprKind::Call(ref func, _) => {
                        if let ExprKind::Path(ref cond_num_path) = func.kind {
                            if INT_TYPES.iter().any(|int_type| match_qpath(cond_num_path, &[int_type, "min_value"])) {
                                print_lint_and_sugg(cx, &var_name, expr);
                            }
                        };
                    },
                    _ => (),
                }
            }
        }
    }
}

fn subtracts_one<'a>(cx: &LateContext<'_>, expr: &Expr<'a>) -> Option<&'a Expr<'a>> {
    match expr.kind {
        ExprKind::AssignOp(ref op1, ref target, ref value) => {
            if_chain! {
                if BinOpKind::Sub == op1.node;
                // Check if literal being subtracted is one
                if let ExprKind::Lit(ref lit1) = value.kind;
                if let LitKind::Int(1, _) = lit1.node;
                then {
                    Some(target)
                } else {
                    None
                }
            }
        },
        ExprKind::Assign(ref target, ref value, _) => {
            if_chain! {
                if let ExprKind::Binary(ref op1, ref left1, ref right1) = value.kind;
                if BinOpKind::Sub == op1.node;

                if SpanlessEq::new(cx).eq_expr(left1, target);

                if let ExprKind::Lit(ref lit1) = right1.kind;
                if let LitKind::Int(1, _) = lit1.node;
                then {
                    Some(target)
                } else {
                    None
                }
            }
        },
        _ => None,
    }
}

fn print_lint_and_sugg(cx: &LateContext<'_>, var_name: &str, expr: &Expr<'_>) {
    span_lint_and_sugg(
        cx,
        IMPLICIT_SATURATING_SUB,
        expr.span,
        "implicitly performing saturating subtraction",
        "try",
        format!("{} = {}.saturating_sub({});", var_name, var_name, '1'),
        Applicability::MachineApplicable,
    );
}
