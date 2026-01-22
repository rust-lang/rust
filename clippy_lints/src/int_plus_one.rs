use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::sugg;
use rustc_ast::ast::{BinOpKind, Expr, ExprKind, LitKind, UnOp};
use rustc_data_structures::packed::Pu128;
use rustc_errors::Applicability;
use rustc_lint::{EarlyContext, EarlyLintPass};
use rustc_session::declare_lint_pass;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `x >= y + 1` or `x - 1 >= y` (and `<=`) in a block
    ///
    /// ### Why is this bad?
    /// Readability -- better to use `> y` instead of `>= y + 1`.
    ///
    /// ### Example
    /// ```no_run
    /// # let x = 1;
    /// # let y = 1;
    /// if x >= y + 1 {}
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// # let x = 1;
    /// # let y = 1;
    /// if x > y {}
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub INT_PLUS_ONE,
    complexity,
    "instead of using `x >= y + 1`, use `x > y`"
}

declare_lint_pass!(IntPlusOne => [INT_PLUS_ONE]);

// cases:
// LeOrGe::Ge
// x >= y + 1
// x - 1 >= y
//
// LeOrGe::Le
// x + 1 <= y
// x <= y - 1

#[derive(Copy, Clone)]
enum LeOrGe {
    Le,
    Ge,
}

impl TryFrom<BinOpKind> for LeOrGe {
    type Error = ();
    fn try_from(value: BinOpKind) -> Result<Self, Self::Error> {
        match value {
            BinOpKind::Le => Ok(Self::Le),
            BinOpKind::Ge => Ok(Self::Ge),
            _ => Err(()),
        }
    }
}

impl IntPlusOne {
    fn is_one(expr: &Expr) -> bool {
        if let ExprKind::Lit(token_lit) = expr.kind
            && matches!(LitKind::from_token_lit(token_lit), Ok(LitKind::Int(Pu128(1), ..)))
        {
            return true;
        }
        false
    }

    fn is_neg_one(expr: &Expr) -> bool {
        if let ExprKind::Unary(UnOp::Neg, expr) = &expr.kind {
            Self::is_one(expr)
        } else {
            false
        }
    }

    /// Checks whether `expr` is `x + 1` or `1 + x`, and if so, returns `x`
    fn as_x_plus_one(expr: &Expr) -> Option<&Expr> {
        if let ExprKind::Binary(op, lhs, rhs) = &expr.kind
            && op.node == BinOpKind::Add
        {
            if Self::is_one(rhs) {
                // x + 1
                return Some(lhs);
            } else if Self::is_one(lhs) {
                // 1 + x
                return Some(rhs);
            }
        }
        None
    }

    /// Checks whether `expr` is `x - 1` or `-1 + x`, and if so, returns `x`
    fn as_x_minus_one(expr: &Expr) -> Option<&Expr> {
        if let ExprKind::Binary(op, lhs, rhs) = &expr.kind {
            if op.node == BinOpKind::Sub && Self::is_one(rhs) {
                // x - 1
                return Some(lhs);
            } else if op.node == BinOpKind::Add && Self::is_neg_one(lhs) {
                // -1 + x
                return Some(rhs);
            }
        }
        None
    }

    fn check_binop<'tcx>(le_or_ge: LeOrGe, lhs: &'tcx Expr, rhs: &'tcx Expr) -> Option<(&'tcx Expr, &'tcx Expr)> {
        match le_or_ge {
            LeOrGe::Ge => {
                // case where `x - 1 >= ...` or `-1 + x >= ...`
                (Self::as_x_minus_one(lhs).map(|new_lhs| (new_lhs, rhs)))
                    // case where `... >= y + 1` or `... >= 1 + y`
                    .or_else(|| Self::as_x_plus_one(rhs).map(|new_rhs| (lhs, new_rhs)))
            },
            LeOrGe::Le => {
                // case where `x + 1 <= ...` or `1 + x <= ...`
                (Self::as_x_plus_one(lhs).map(|new_lhs| (new_lhs, rhs)))
                    // case where `... <= y - 1` or `... <= -1 + y`
                    .or_else(|| Self::as_x_minus_one(rhs).map(|new_rhs| (lhs, new_rhs)))
            },
        }
    }

    fn emit_warning(cx: &EarlyContext<'_>, expr: &Expr, new_lhs: &Expr, le_or_ge: LeOrGe, new_rhs: &Expr) {
        span_lint_and_then(
            cx,
            INT_PLUS_ONE,
            expr.span,
            "unnecessary `>= y + 1` or `x - 1 >=`",
            |diag| {
                let mut app = Applicability::MachineApplicable;
                let ctxt = expr.span.ctxt();
                let new_lhs = sugg::Sugg::ast(cx, new_lhs, "_", ctxt, &mut app);
                let new_rhs = sugg::Sugg::ast(cx, new_rhs, "_", ctxt, &mut app);
                let new_binop = match le_or_ge {
                    LeOrGe::Ge => BinOpKind::Gt,
                    LeOrGe::Le => BinOpKind::Lt,
                };
                let rec = sugg::make_binop(new_binop, &new_lhs, &new_rhs);
                diag.span_suggestion(expr.span, "change it to", rec, app);
            },
        );
    }
}

impl EarlyLintPass for IntPlusOne {
    fn check_expr(&mut self, cx: &EarlyContext<'_>, expr: &Expr) {
        if let ExprKind::Binary(binop, lhs, rhs) = &expr.kind
            && let Ok(le_or_ge) = LeOrGe::try_from(binop.node)
            && let Some((new_lhs, new_rhs)) = Self::check_binop(le_or_ge, lhs, rhs)
        {
            Self::emit_warning(cx, expr, new_lhs, le_or_ge, new_rhs);
        }
    }
}
