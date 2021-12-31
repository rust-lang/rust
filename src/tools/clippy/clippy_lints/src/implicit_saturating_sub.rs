use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::{higher, peel_blocks_with_stmt, SpanlessEq};
use if_chain::if_chain;
use rustc_ast::ast::LitKind;
use rustc_errors::Applicability;
use rustc_hir::{lang_items::LangItem, BinOpKind, Expr, ExprKind, QPath};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for implicit saturating subtraction.
    ///
    /// ### Why is this bad?
    /// Simplicity and readability. Instead we can easily use an builtin function.
    ///
    /// ### Example
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
    #[clippy::version = "1.44.0"]
    pub IMPLICIT_SATURATING_SUB,
    pedantic,
    "Perform saturating subtraction instead of implicitly checking lower bound of data type"
}

declare_lint_pass!(ImplicitSaturatingSub => [IMPLICIT_SATURATING_SUB]);

impl<'tcx> LateLintPass<'tcx> for ImplicitSaturatingSub {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) {
        if expr.span.from_expansion() {
            return;
        }
        if_chain! {
            if let Some(higher::If { cond, then, r#else: None }) = higher::If::hir(expr);

            // Check if the conditional expression is a binary operation
            if let ExprKind::Binary(ref cond_op, cond_left, cond_right) = cond.kind;

            // Ensure that the binary operator is >, != and <
            if BinOpKind::Ne == cond_op.node || BinOpKind::Gt == cond_op.node || BinOpKind::Lt == cond_op.node;

            // Check if assign operation is done
            if let Some(target) = subtracts_one(cx, then);

            // Extracting out the variable name
            if let ExprKind::Path(QPath::Resolved(_, ares_path)) = target.kind;

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
                const INT_TYPES: [LangItem; 5] = [
                    LangItem::I8,
                    LangItem::I16,
                    LangItem::I32,
                    LangItem::I64,
                    LangItem::Isize
                ];

                match cond_num_val.kind {
                    ExprKind::Lit(ref cond_lit) => {
                        // Check if the constant is zero
                        if let LitKind::Int(0, _) = cond_lit.node {
                            if cx.typeck_results().expr_ty(cond_left).is_signed() {
                            } else {
                                print_lint_and_sugg(cx, var_name, expr);
                            };
                        }
                    },
                    ExprKind::Path(QPath::TypeRelative(_, name)) => {
                        if_chain! {
                            if name.ident.as_str() == "MIN";
                            if let Some(const_id) = cx.typeck_results().type_dependent_def_id(cond_num_val.hir_id);
                            if let Some(impl_id) = cx.tcx.impl_of_method(const_id);
                            let mut int_ids = INT_TYPES.iter().filter_map(|&ty| cx.tcx.lang_items().require(ty).ok());
                            if int_ids.any(|int_id| int_id == impl_id);
                            then {
                                print_lint_and_sugg(cx, var_name, expr)
                            }
                        }
                    },
                    ExprKind::Call(func, []) => {
                        if_chain! {
                            if let ExprKind::Path(QPath::TypeRelative(_, name)) = func.kind;
                            if name.ident.as_str() == "min_value";
                            if let Some(func_id) = cx.typeck_results().type_dependent_def_id(func.hir_id);
                            if let Some(impl_id) = cx.tcx.impl_of_method(func_id);
                            let mut int_ids = INT_TYPES.iter().filter_map(|&ty| cx.tcx.lang_items().require(ty).ok());
                            if int_ids.any(|int_id| int_id == impl_id);
                            then {
                                print_lint_and_sugg(cx, var_name, expr)
                            }
                        }
                    },
                    _ => (),
                }
            }
        }
    }
}

fn subtracts_one<'a>(cx: &LateContext<'_>, expr: &'a Expr<'a>) -> Option<&'a Expr<'a>> {
    match peel_blocks_with_stmt(expr).kind {
        ExprKind::AssignOp(ref op1, target, value) => {
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
        ExprKind::Assign(target, value, _) => {
            if_chain! {
                if let ExprKind::Binary(ref op1, left1, right1) = value.kind;
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
