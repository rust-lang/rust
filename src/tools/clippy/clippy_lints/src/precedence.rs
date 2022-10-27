use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet_with_applicability;
use if_chain::if_chain;
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
    /// ### What it does
    /// Checks for operations where precedence may be unclear
    /// and suggests to add parentheses. Currently it catches the following:
    /// * mixed usage of arithmetic and bit shifting/combining operators without
    /// parentheses
    /// * a "negative" numeric literal (which is really a unary `-` followed by a
    /// numeric literal)
    ///   followed by a method call
    ///
    /// ### Why is this bad?
    /// Not everyone knows the precedence of those operators by
    /// heart, so expressions like these may trip others trying to reason about the
    /// code.
    ///
    /// ### Example
    /// * `1 << 2 + 3` equals 32, while `(1 << 2) + 3` equals 7
    /// * `-1i32.abs()` equals -1, while `(-1i32).abs()` equals 1
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

        if let ExprKind::Unary(UnOp::Neg, operand) = &expr.kind {
            let mut arg = operand;

            let mut all_odd = true;
            while let ExprKind::MethodCall(path_segment, args, _) = &arg.kind {
                let path_segment_str = path_segment.ident.name.as_str();
                all_odd &= ALLOWED_ODD_FUNCTIONS
                    .iter()
                    .any(|odd_function| **odd_function == *path_segment_str);
                arg = args.first().expect("A method always has a receiver.");
            }

            if_chain! {
                if !all_odd;
                if let ExprKind::Lit(lit) = &arg.kind;
                if let LitKind::Int(..) | LitKind::Float(..) = &lit.kind;
                then {
                    let mut applicability = Applicability::MachineApplicable;
                    span_lint_and_sugg(
                        cx,
                        PRECEDENCE,
                        expr.span,
                        "unary minus has lower precedence than method call",
                        "consider adding parentheses to clarify your intent",
                        format!(
                            "-({})",
                            snippet_with_applicability(cx, operand.span, "..", &mut applicability)
                        ),
                        applicability,
                    );
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
