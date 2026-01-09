use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::SpanRangeExt;
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
            && let Ok(LitKind::Int(Pu128(1), ..)) = LitKind::from_token_lit(token_lit)
        {
            return true;
        }
        false
    }

    fn is_neg_one(expr: &Expr) -> bool {
        if let ExprKind::Unary(UnOp::Neg, expr) = &expr.kind
            && Self::is_one(expr)
        {
            true
        } else {
            false
        }
    }

    fn check_binop(cx: &EarlyContext<'_>, le_or_ge: LeOrGe, lhs: &Expr, rhs: &Expr) -> Option<String> {
        match (le_or_ge, &lhs.kind, &rhs.kind) {
            // case where `x - 1 >= ...` or `-1 + x >= ...`
            (LeOrGe::Ge, ExprKind::Binary(lhskind, lhslhs, lhsrhs), _) => {
                match lhskind.node {
                    // `-1 + x`
                    BinOpKind::Add if Self::is_neg_one(lhslhs) => {
                        Self::generate_recommendation(cx, le_or_ge, lhsrhs, rhs)
                    },
                    // `x - 1`
                    BinOpKind::Sub if Self::is_one(lhsrhs) => Self::generate_recommendation(cx, le_or_ge, lhslhs, rhs),
                    _ => None,
                }
            },
            // case where `... >= y + 1` or `... >= 1 + y`
            (LeOrGe::Ge, _, ExprKind::Binary(rhskind, rhslhs, rhsrhs)) if rhskind.node == BinOpKind::Add => {
                // `y + 1` and `1 + y`
                if Self::is_one(rhslhs) {
                    Self::generate_recommendation(cx, le_or_ge, lhs, rhsrhs)
                } else if Self::is_one(rhsrhs) {
                    Self::generate_recommendation(cx, le_or_ge, lhs, rhslhs)
                } else {
                    None
                }
            },
            // case where `x + 1 <= ...` or `1 + x <= ...`
            (LeOrGe::Le, ExprKind::Binary(lhskind, lhslhs, lhsrhs), _) if lhskind.node == BinOpKind::Add => {
                // `1 + x` and `x + 1`
                if Self::is_one(lhslhs) {
                    Self::generate_recommendation(cx, le_or_ge, lhsrhs, rhs)
                } else if Self::is_one(lhsrhs) {
                    Self::generate_recommendation(cx, le_or_ge, lhslhs, rhs)
                } else {
                    None
                }
            },
            // case where `... <= y - 1` or `... <= -1 + y`
            (LeOrGe::Le, _, ExprKind::Binary(rhskind, rhslhs, rhsrhs)) => {
                match rhskind.node {
                    // `-1 + y`
                    BinOpKind::Add if Self::is_neg_one(rhslhs) => {
                        Self::generate_recommendation(cx, le_or_ge, lhs, rhsrhs)
                    },
                    // `y - 1`
                    BinOpKind::Sub if Self::is_one(rhsrhs) => Self::generate_recommendation(cx, le_or_ge, lhs, rhslhs),
                    _ => None,
                }
            },
            _ => None,
        }
    }

    fn generate_recommendation(
        cx: &EarlyContext<'_>,
        le_or_ge: LeOrGe,
        node: &Expr,
        other_side: &Expr,
    ) -> Option<String> {
        let binop_string = match le_or_ge {
            LeOrGe::Ge => ">",
            LeOrGe::Le => "<",
        };
        if let Some(snippet) = node.span.get_source_text(cx)
            && let Some(other_side_snippet) = other_side.span.get_source_text(cx)
        {
            return Some(format!("{snippet} {binop_string} {other_side_snippet}"));
        }
        None
    }

    fn emit_warning(cx: &EarlyContext<'_>, block: &Expr, recommendation: String) {
        span_lint_and_sugg(
            cx,
            INT_PLUS_ONE,
            block.span,
            "unnecessary `>= y + 1` or `x - 1 >=`",
            "change it to",
            recommendation,
            Applicability::MachineApplicable, // snippet
        );
    }
}

impl EarlyLintPass for IntPlusOne {
    fn check_expr(&mut self, cx: &EarlyContext<'_>, expr: &Expr) {
        if let ExprKind::Binary(binop, lhs, rhs) = &expr.kind
            && let Ok(le_or_ge) = LeOrGe::try_from(binop.node)
            && let Some(rec) = Self::check_binop(cx, le_or_ge, lhs, rhs)
        {
            Self::emit_warning(cx, expr, rec);
        }
    }
}
