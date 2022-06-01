use rustc_hir::{Body, Expr, ExprKind, UnOp};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_tool_lint, impl_lint_pass};

mod absurd_extreme_comparisons;
mod numeric_arithmetic;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for comparisons where one side of the relation is
    /// either the minimum or maximum value for its type and warns if it involves a
    /// case that is always true or always false. Only integer and boolean types are
    /// checked.
    ///
    /// ### Why is this bad?
    /// An expression like `min <= x` may misleadingly imply
    /// that it is possible for `x` to be less than the minimum. Expressions like
    /// `max < x` are probably mistakes.
    ///
    /// ### Known problems
    /// For `usize` the size of the current compile target will
    /// be assumed (e.g., 64 bits on 64 bit systems). This means code that uses such
    /// a comparison to detect target pointer width will trigger this lint. One can
    /// use `mem::sizeof` and compare its value or conditional compilation
    /// attributes
    /// like `#[cfg(target_pointer_width = "64")] ..` instead.
    ///
    /// ### Example
    /// ```rust
    /// let vec: Vec<isize> = Vec::new();
    /// if vec.len() <= 0 {}
    /// if 100 > i32::MAX {}
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub ABSURD_EXTREME_COMPARISONS,
    correctness,
    "a comparison with a maximum or minimum value that is always true or false"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for integer arithmetic operations which could overflow or panic.
    ///
    /// Specifically, checks for any operators (`+`, `-`, `*`, `<<`, etc) which are capable
    /// of overflowing according to the [Rust
    /// Reference](https://doc.rust-lang.org/reference/expressions/operator-expr.html#overflow),
    /// or which can panic (`/`, `%`). No bounds analysis or sophisticated reasoning is
    /// attempted.
    ///
    /// ### Why is this bad?
    /// Integer overflow will trigger a panic in debug builds or will wrap in
    /// release mode. Division by zero will cause a panic in either mode. In some applications one
    /// wants explicitly checked, wrapping or saturating arithmetic.
    ///
    /// ### Example
    /// ```rust
    /// # let a = 0;
    /// a + 1;
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub INTEGER_ARITHMETIC,
    restriction,
    "any integer arithmetic expression which could overflow or panic"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for float arithmetic.
    ///
    /// ### Why is this bad?
    /// For some embedded systems or kernel development, it
    /// can be useful to rule out floating-point numbers.
    ///
    /// ### Example
    /// ```rust
    /// # let a = 0.0;
    /// a + 1.0;
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub FLOAT_ARITHMETIC,
    restriction,
    "any floating-point arithmetic statement"
}

#[derive(Default)]
pub struct Operators {
    arithmetic_context: numeric_arithmetic::Context,
}
impl_lint_pass!(Operators => [
    ABSURD_EXTREME_COMPARISONS,
    INTEGER_ARITHMETIC,
    FLOAT_ARITHMETIC,
]);
impl<'tcx> LateLintPass<'tcx> for Operators {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, e: &'tcx Expr<'_>) {
        match e.kind {
            ExprKind::Binary(op, lhs, rhs) => {
                if !e.span.from_expansion() {
                    absurd_extreme_comparisons::check(cx, e, op.node, lhs, rhs);
                }
                self.arithmetic_context.check_binary(cx, e, op.node, lhs, rhs);
            },
            ExprKind::AssignOp(op, lhs, rhs) => {
                self.arithmetic_context.check_binary(cx, e, op.node, lhs, rhs);
            },
            ExprKind::Unary(op, arg) => {
                if op == UnOp::Neg {
                    self.arithmetic_context.check_negate(cx, e, arg);
                }
            },
            _ => (),
        }
    }

    fn check_expr_post(&mut self, _: &LateContext<'_>, e: &Expr<'_>) {
        self.arithmetic_context.expr_post(e.hir_id);
    }

    fn check_body(&mut self, cx: &LateContext<'tcx>, b: &'tcx Body<'_>) {
        self.arithmetic_context.enter_body(cx, b);
    }

    fn check_body_post(&mut self, cx: &LateContext<'tcx>, b: &'tcx Body<'_>) {
        self.arithmetic_context.body_post(cx, b);
    }
}
