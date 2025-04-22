use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::SpanRangeExt;
use rustc_ast::ast::{BinOpKind, Expr, ExprKind, LitKind};
use rustc_ast::token;
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
// BinOpKind::Ge
// x >= y + 1
// x - 1 >= y
//
// BinOpKind::Le
// x + 1 <= y
// x <= y - 1

#[derive(Copy, Clone)]
enum Side {
    Lhs,
    Rhs,
}

impl IntPlusOne {
    #[expect(clippy::cast_sign_loss)]
    fn check_lit(token_lit: token::Lit, target_value: i128) -> bool {
        if let Ok(LitKind::Int(value, ..)) = LitKind::from_token_lit(token_lit) {
            return value == (target_value as u128);
        }
        false
    }

    fn check_binop(cx: &EarlyContext<'_>, binop: BinOpKind, lhs: &Expr, rhs: &Expr) -> Option<String> {
        match (binop, &lhs.kind, &rhs.kind) {
            // case where `x - 1 >= ...` or `-1 + x >= ...`
            (BinOpKind::Ge, ExprKind::Binary(lhskind, lhslhs, lhsrhs), _) => {
                match (lhskind.node, &lhslhs.kind, &lhsrhs.kind) {
                    // `-1 + x`
                    (BinOpKind::Add, ExprKind::Lit(lit), _) if Self::check_lit(*lit, -1) => {
                        Self::generate_recommendation(cx, binop, lhsrhs, rhs, Side::Lhs)
                    },
                    // `x - 1`
                    (BinOpKind::Sub, _, ExprKind::Lit(lit)) if Self::check_lit(*lit, 1) => {
                        Self::generate_recommendation(cx, binop, lhslhs, rhs, Side::Lhs)
                    },
                    _ => None,
                }
            },
            // case where `... >= y + 1` or `... >= 1 + y`
            (BinOpKind::Ge, _, ExprKind::Binary(rhskind, rhslhs, rhsrhs)) if rhskind.node == BinOpKind::Add => {
                match (&rhslhs.kind, &rhsrhs.kind) {
                    // `y + 1` and `1 + y`
                    (ExprKind::Lit(lit), _) if Self::check_lit(*lit, 1) => {
                        Self::generate_recommendation(cx, binop, rhsrhs, lhs, Side::Rhs)
                    },
                    (_, ExprKind::Lit(lit)) if Self::check_lit(*lit, 1) => {
                        Self::generate_recommendation(cx, binop, rhslhs, lhs, Side::Rhs)
                    },
                    _ => None,
                }
            },
            // case where `x + 1 <= ...` or `1 + x <= ...`
            (BinOpKind::Le, ExprKind::Binary(lhskind, lhslhs, lhsrhs), _) if lhskind.node == BinOpKind::Add => {
                match (&lhslhs.kind, &lhsrhs.kind) {
                    // `1 + x` and `x + 1`
                    (ExprKind::Lit(lit), _) if Self::check_lit(*lit, 1) => {
                        Self::generate_recommendation(cx, binop, lhsrhs, rhs, Side::Lhs)
                    },
                    (_, ExprKind::Lit(lit)) if Self::check_lit(*lit, 1) => {
                        Self::generate_recommendation(cx, binop, lhslhs, rhs, Side::Lhs)
                    },
                    _ => None,
                }
            },
            // case where `... >= y - 1` or `... >= -1 + y`
            (BinOpKind::Le, _, ExprKind::Binary(rhskind, rhslhs, rhsrhs)) => {
                match (rhskind.node, &rhslhs.kind, &rhsrhs.kind) {
                    // `-1 + y`
                    (BinOpKind::Add, ExprKind::Lit(lit), _) if Self::check_lit(*lit, -1) => {
                        Self::generate_recommendation(cx, binop, rhsrhs, lhs, Side::Rhs)
                    },
                    // `y - 1`
                    (BinOpKind::Sub, _, ExprKind::Lit(lit)) if Self::check_lit(*lit, 1) => {
                        Self::generate_recommendation(cx, binop, rhslhs, lhs, Side::Rhs)
                    },
                    _ => None,
                }
            },
            _ => None,
        }
    }

    fn generate_recommendation(
        cx: &EarlyContext<'_>,
        binop: BinOpKind,
        node: &Expr,
        other_side: &Expr,
        side: Side,
    ) -> Option<String> {
        let binop_string = match binop {
            BinOpKind::Ge => ">",
            BinOpKind::Le => "<",
            _ => return None,
        };
        if let Some(snippet) = node.span.get_source_text(cx)
            && let Some(other_side_snippet) = other_side.span.get_source_text(cx)
        {
            let rec = match side {
                Side::Lhs => Some(format!("{snippet} {binop_string} {other_side_snippet}")),
                Side::Rhs => Some(format!("{other_side_snippet} {binop_string} {snippet}")),
            };
            return rec;
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
    fn check_expr(&mut self, cx: &EarlyContext<'_>, item: &Expr) {
        if let ExprKind::Binary(ref kind, ref lhs, ref rhs) = item.kind
            && let Some(rec) = Self::check_binop(cx, kind.node, lhs, rhs)
        {
            Self::emit_warning(cx, item, rec);
        }
    }
}
