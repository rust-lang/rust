use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet_with_applicability;
use rustc_ast::ast::BinOpKind::{Add, BitAnd, BitOr, BitXor, Div, Mul, Rem, Shl, Shr, Sub};
use rustc_ast::ast::{BinOpKind, Expr, ExprKind};
use rustc_errors::Applicability;
use rustc_lint::{EarlyContext, EarlyLintPass};
use rustc_session::declare_lint_pass;
use rustc_span::source_map::Spanned;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for operations where precedence may be unclear
    /// and suggests to add parentheses. Currently it catches the following:
    /// * mixed usage of arithmetic and bit shifting/combining operators without
    /// parentheses
    /// * mixed usage of bitmasking and bit shifting operators without parentheses
    ///
    /// ### Why is this bad?
    /// Not everyone knows the precedence of those operators by
    /// heart, so expressions like these may trip others trying to reason about the
    /// code.
    ///
    /// ### Example
    /// * `1 << 2 + 3` equals 32, while `(1 << 2) + 3` equals 7
    /// * `0x2345 & 0xF000 >> 12` equals 5, while `(0x2345 & 0xF000) >> 12` equals 2
    #[clippy::version = "pre 1.29.0"]
    pub PRECEDENCE,
    complexity,
    "operations where precedence may be unclear"
}

declare_lint_pass!(Precedence => [PRECEDENCE]);

impl EarlyLintPass for Precedence {
    fn check_expr(&mut self, cx: &EarlyContext<'_>, expr: &Expr) {
        if expr.span.from_expansion() {
            return;
        }

        if let ExprKind::Binary(Spanned { node: op, .. }, ref left, ref right) = expr.kind {
            let span_sugg = |expr: &Expr, sugg, appl| {
                span_lint_and_sugg(
                    cx,
                    PRECEDENCE,
                    expr.span,
                    "operator precedence can trip the unwary",
                    "consider parenthesizing your expression",
                    sugg,
                    appl,
                );
            };

            if !is_bit_op(op) {
                return;
            }
            let mut applicability = Applicability::MachineApplicable;
            match (op, get_bin_opt(left), get_bin_opt(right)) {
                (
                    BitAnd | BitOr | BitXor,
                    Some(Shl | Shr | Add | Div | Mul | Rem | Sub),
                    Some(Shl | Shr | Add | Div | Mul | Rem | Sub),
                )
                | (Shl | Shr, Some(Add | Div | Mul | Rem | Sub), Some(Add | Div | Mul | Rem | Sub)) => {
                    let sugg = format!(
                        "({}) {} ({})",
                        snippet_with_applicability(cx, left.span, "..", &mut applicability),
                        op.as_str(),
                        snippet_with_applicability(cx, right.span, "..", &mut applicability)
                    );
                    span_sugg(expr, sugg, applicability);
                },
                (BitAnd | BitOr | BitXor, Some(Shl | Shr | Add | Div | Mul | Rem | Sub), _)
                | (Shl | Shr, Some(Add | Div | Mul | Rem | Sub), _) => {
                    let sugg = format!(
                        "({}) {} {}",
                        snippet_with_applicability(cx, left.span, "..", &mut applicability),
                        op.as_str(),
                        snippet_with_applicability(cx, right.span, "..", &mut applicability)
                    );
                    span_sugg(expr, sugg, applicability);
                },
                (BitAnd | BitOr | BitXor, _, Some(Shl | Shr | Add | Div | Mul | Rem | Sub))
                | (Shl | Shr, _, Some(Add | Div | Mul | Rem | Sub)) => {
                    let sugg = format!(
                        "{} {} ({})",
                        snippet_with_applicability(cx, left.span, "..", &mut applicability),
                        op.as_str(),
                        snippet_with_applicability(cx, right.span, "..", &mut applicability)
                    );
                    span_sugg(expr, sugg, applicability);
                },
                _ => (),
            }
        }
    }
}

fn get_bin_opt(expr: &Expr) -> Option<BinOpKind> {
    match expr.kind {
        ExprKind::Binary(Spanned { node: op, .. }, _, _) => Some(op),
        _ => None,
    }
}

#[must_use]
fn is_bit_op(op: BinOpKind) -> bool {
    matches!(op, BitXor | BitAnd | BitOr | Shl | Shr)
}
