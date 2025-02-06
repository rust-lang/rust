use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet_with_applicability;
use rustc_ast::ast::BinOpKind::{Add, BitAnd, BitOr, BitXor, Div, Mul, Rem, Shl, Shr, Sub};
use rustc_ast::ast::{BinOpKind, Expr, ExprKind};
use rustc_errors::Applicability;
use rustc_lint::{EarlyContext, EarlyLintPass, Lint};
use rustc_session::declare_lint_pass;
use rustc_span::source_map::Spanned;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for operations where precedence may be unclear and suggests to add parentheses.
    /// It catches a mixed usage of arithmetic and bit shifting/combining operators without parentheses
    ///
    /// ### Why is this bad?
    /// Not everyone knows the precedence of those operators by
    /// heart, so expressions like these may trip others trying to reason about the
    /// code.
    ///
    /// ### Example
    /// `1 << 2 + 3` equals 32, while `(1 << 2) + 3` equals 7
    #[clippy::version = "pre 1.29.0"]
    pub PRECEDENCE,
    complexity,
    "operations where precedence may be unclear"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for bit shifting operations combined with bit masking/combining operators
    /// and suggest using parentheses.
    ///
    /// ### Why restrict this?
    /// Not everyone knows the precedence of those operators by
    /// heart, so expressions like these may trip others trying to reason about the
    /// code.
    ///
    /// ### Example
    /// `0x2345 & 0xF000 >> 12` equals 5, while `(0x2345 & 0xF000) >> 12` equals 2
    #[clippy::version = "1.86.0"]
    pub PRECEDENCE_BITS,
    restriction,
    "operations mixing bit shifting with bit combining/masking"
}

declare_lint_pass!(Precedence => [PRECEDENCE, PRECEDENCE_BITS]);

impl EarlyLintPass for Precedence {
    fn check_expr(&mut self, cx: &EarlyContext<'_>, expr: &Expr) {
        if expr.span.from_expansion() {
            return;
        }

        if let ExprKind::Binary(Spanned { node: op, .. }, ref left, ref right) = expr.kind {
            let span_sugg = |lint: &'static Lint, expr: &Expr, sugg, appl| {
                span_lint_and_sugg(
                    cx,
                    lint,
                    expr.span,
                    "operator precedence might not be obvious",
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
                    Some(left_op @ (Shl | Shr | Add | Div | Mul | Rem | Sub)),
                    Some(right_op @ (Shl | Shr | Add | Div | Mul | Rem | Sub)),
                )
                | (
                    Shl | Shr,
                    Some(left_op @ (Add | Div | Mul | Rem | Sub)),
                    Some(right_op @ (Add | Div | Mul | Rem | Sub)),
                ) => {
                    let sugg = format!(
                        "({}) {} ({})",
                        snippet_with_applicability(cx, left.span, "..", &mut applicability),
                        op.as_str(),
                        snippet_with_applicability(cx, right.span, "..", &mut applicability)
                    );
                    span_sugg(lint_for(&[op, left_op, right_op]), expr, sugg, applicability);
                },
                (BitAnd | BitOr | BitXor, Some(side_op @ (Shl | Shr | Add | Div | Mul | Rem | Sub)), _)
                | (Shl | Shr, Some(side_op @ (Add | Div | Mul | Rem | Sub)), _) => {
                    let sugg = format!(
                        "({}) {} {}",
                        snippet_with_applicability(cx, left.span, "..", &mut applicability),
                        op.as_str(),
                        snippet_with_applicability(cx, right.span, "..", &mut applicability)
                    );
                    span_sugg(lint_for(&[op, side_op]), expr, sugg, applicability);
                },
                (BitAnd | BitOr | BitXor, _, Some(side_op @ (Shl | Shr | Add | Div | Mul | Rem | Sub)))
                | (Shl | Shr, _, Some(side_op @ (Add | Div | Mul | Rem | Sub))) => {
                    let sugg = format!(
                        "{} {} ({})",
                        snippet_with_applicability(cx, left.span, "..", &mut applicability),
                        op.as_str(),
                        snippet_with_applicability(cx, right.span, "..", &mut applicability)
                    );
                    span_sugg(lint_for(&[op, side_op]), expr, sugg, applicability);
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

fn lint_for(ops: &[BinOpKind]) -> &'static Lint {
    if ops.iter().all(|op| is_bit_op(*op)) {
        PRECEDENCE_BITS
    } else {
        PRECEDENCE
    }
}
