use crate::utils::{snippet_with_applicability, span_lint_and_sugg};
use rustc_ast::ast::{BinOpKind, Expr, ExprKind, LitKind, UnOp};
use rustc_errors::Applicability;
use rustc_lint::{EarlyContext, EarlyLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::source_map::Spanned;

const ALLOWED_ODD_FUNCTIONS: [&str; 14] = [
    "asin",
    "asinh",
    "atan",
    "atanh",
    "cbrt",
    "fract",
    "round",
    "signum",
    "sin",
    "sinh",
    "tan",
    "tanh",
    "to_degrees",
    "to_radians",
];

declare_clippy_lint! {
    /// **What it does:** Checks for operations where precedence may be unclear
    /// and suggests to add parentheses. Currently it catches the following:
    /// * mixed usage of arithmetic and bit shifting/combining operators without
    /// parentheses
    /// * a "negative" numeric literal (which is really a unary `-` followed by a
    /// numeric literal)
    ///   followed by a method call
    ///
    /// **Why is this bad?** Not everyone knows the precedence of those operators by
    /// heart, so expressions like these may trip others trying to reason about the
    /// code.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// * `1 << 2 + 3` equals 32, while `(1 << 2) + 3` equals 7
    /// * `-1i32.abs()` equals -1, while `(-1i32).abs()` equals 1
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
            match (is_arith_expr(left), is_arith_expr(right)) {
                (true, true) => {
                    let sugg = format!(
                        "({}) {} ({})",
                        snippet_with_applicability(cx, left.span, "..", &mut applicability),
                        op.to_string(),
                        snippet_with_applicability(cx, right.span, "..", &mut applicability)
                    );
                    span_sugg(expr, sugg, applicability);
                },
                (true, false) => {
                    let sugg = format!(
                        "({}) {} {}",
                        snippet_with_applicability(cx, left.span, "..", &mut applicability),
                        op.to_string(),
                        snippet_with_applicability(cx, right.span, "..", &mut applicability)
                    );
                    span_sugg(expr, sugg, applicability);
                },
                (false, true) => {
                    let sugg = format!(
                        "{} {} ({})",
                        snippet_with_applicability(cx, left.span, "..", &mut applicability),
                        op.to_string(),
                        snippet_with_applicability(cx, right.span, "..", &mut applicability)
                    );
                    span_sugg(expr, sugg, applicability);
                },
                (false, false) => (),
            }
        }

        if let ExprKind::Unary(UnOp::Neg, ref rhs) = expr.kind {
            if let ExprKind::MethodCall(ref path_segment, ref args, _) = rhs.kind {
                let path_segment_str = path_segment.ident.name.as_str();
                if let Some(slf) = args.first() {
                    if let ExprKind::Lit(ref lit) = slf.kind {
                        match lit.kind {
                            LitKind::Int(..) | LitKind::Float(..) => {
                                if ALLOWED_ODD_FUNCTIONS
                                    .iter()
                                    .any(|odd_function| **odd_function == *path_segment_str)
                                {
                                    return;
                                }
                                let mut applicability = Applicability::MachineApplicable;
                                span_lint_and_sugg(
                                    cx,
                                    PRECEDENCE,
                                    expr.span,
                                    "unary minus has lower precedence than method call",
                                    "consider adding parentheses to clarify your intent",
                                    format!(
                                        "-({})",
                                        snippet_with_applicability(cx, rhs.span, "..", &mut applicability)
                                    ),
                                    applicability,
                                );
                            },
                            _ => (),
                        }
                    }
                }
            }
        }
    }
}

fn is_arith_expr(expr: &Expr) -> bool {
    match expr.kind {
        ExprKind::Binary(Spanned { node: op, .. }, _, _) => is_arith_op(op),
        _ => false,
    }
}

#[must_use]
fn is_bit_op(op: BinOpKind) -> bool {
    use rustc_ast::ast::BinOpKind::{BitAnd, BitOr, BitXor, Shl, Shr};
    matches!(op, BitXor | BitAnd | BitOr | Shl | Shr)
}

#[must_use]
fn is_arith_op(op: BinOpKind) -> bool {
    use rustc_ast::ast::BinOpKind::{Add, Div, Mul, Rem, Sub};
    matches!(op, Add | Sub | Mul | Div | Rem)
}
