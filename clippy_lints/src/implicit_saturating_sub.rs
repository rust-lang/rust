use crate::utils::{higher, in_macro, span_lint_and_sugg};
use if_chain::if_chain;
use rustc_ast::ast::LitKind;
use rustc_errors::Applicability;
use rustc_hir::{BinOpKind, Expr, ExprKind, QPath, StmtKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// **What it does:** Checks for implicit saturating subtraction.
    ///
    /// **Why is this bad?** Simplicity and readability. Instead we can easily use an inbuilt function.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    ///
    /// ```rust
    /// let end = 10;
    /// let start = 5;
    ///
    /// let mut i = end - start;
    ///
    /// // Bad
    /// if i != 0 {
    ///     i -= 1;
    /// }
    /// ```
    /// Use instead:
    /// ```rust
    /// let end = 10;
    /// let start = 5;
    ///
    /// let mut i = end - start;
    ///
    /// // Good
    /// i.saturating_sub(1);
    /// ```
    pub IMPLICIT_SATURATING_SUB,
    pedantic,
    "Perform saturating subtraction instead of implicitly checking lower bound of data type"
}

declare_lint_pass!(ImplicitSaturatingSub => [IMPLICIT_SATURATING_SUB]);

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for ImplicitSaturatingSub {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr<'tcx>) {
        if in_macro(expr.span) {
            return;
        }
        if_chain! {
            if let Some((ref cond, ref then, None)) = higher::if_block(&expr);
            // Check if the conditional expression is a binary operation
            if let ExprKind::Binary(ref op, ref left, ref right) = cond.kind;
            // Ensure that the binary operator is > or !=
            if BinOpKind::Ne == op.node || BinOpKind::Gt == op.node;
            if let ExprKind::Path(ref cond_path) = left.kind;
            // Get the literal on the right hand side
            if let ExprKind::Lit(ref lit) = right.kind;
            if let LitKind::Int(0, _) = lit.node;
            // Check if the true condition block has only one statement
            if let ExprKind::Block(ref block, _) = then.kind;
            if block.stmts.len() == 1;
            // Check if assign operation is done
            if let StmtKind::Semi(ref e) = block.stmts[0].kind;
            if let ExprKind::AssignOp(ref op1, ref target, ref value) = e.kind;
            if BinOpKind::Sub == op1.node;
            if let ExprKind::Path(ref assign_path) = target.kind;
            // Check if the variable in the condition and assignment statement are the same
            if let (QPath::Resolved(_, ref cres_path), QPath::Resolved(_, ref ares_path)) = (cond_path, assign_path);
            if cres_path.res == ares_path.res;
            if let ExprKind::Lit(ref lit1) = value.kind;
            if let LitKind::Int(assign_lit, _) = lit1.node;
            then {
                // Get the variable name
                let var_name = ares_path.segments[0].ident.name.as_str();
                let applicability = Applicability::MaybeIncorrect;
                span_lint_and_sugg(
                    cx,
                    IMPLICIT_SATURATING_SUB,
                    expr.span,
                    "Implicitly performing saturating subtraction",
                    "try",
                    format!("{}.saturating_sub({});", var_name, assign_lit.to_string()),
                    applicability
                );
            }
        }
    }
}
