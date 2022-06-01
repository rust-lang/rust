use rustc_hir::{Body, Expr, ExprKind, UnOp};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_tool_lint, impl_lint_pass};

mod absurd_extreme_comparisons;
mod assign_op_pattern;
mod misrefactored_assign_op;
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

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `a = a op b` or `a = b commutative_op a`
    /// patterns.
    ///
    /// ### Why is this bad?
    /// These can be written as the shorter `a op= b`.
    ///
    /// ### Known problems
    /// While forbidden by the spec, `OpAssign` traits may have
    /// implementations that differ from the regular `Op` impl.
    ///
    /// ### Example
    /// ```rust
    /// let mut a = 5;
    /// let b = 0;
    /// // ...
    ///
    /// a = a + b;
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// let mut a = 5;
    /// let b = 0;
    /// // ...
    ///
    /// a += b;
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub ASSIGN_OP_PATTERN,
    style,
    "assigning the result of an operation on a variable to that same variable"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `a op= a op b` or `a op= b op a` patterns.
    ///
    /// ### Why is this bad?
    /// Most likely these are bugs where one meant to write `a
    /// op= b`.
    ///
    /// ### Known problems
    /// Clippy cannot know for sure if `a op= a op b` should have
    /// been `a = a op a op b` or `a = a op b`/`a op= b`. Therefore, it suggests both.
    /// If `a op= a op b` is really the correct behavior it should be
    /// written as `a = a op a op b` as it's less confusing.
    ///
    /// ### Example
    /// ```rust
    /// let mut a = 5;
    /// let b = 2;
    /// // ...
    /// a += a + b;
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub MISREFACTORED_ASSIGN_OP,
    suspicious,
    "having a variable on both sides of an assign op"
}

#[derive(Default)]
pub struct Operators {
    arithmetic_context: numeric_arithmetic::Context,
}
impl_lint_pass!(Operators => [
    ABSURD_EXTREME_COMPARISONS,
    INTEGER_ARITHMETIC,
    FLOAT_ARITHMETIC,
    ASSIGN_OP_PATTERN,
    MISREFACTORED_ASSIGN_OP,
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
                misrefactored_assign_op::check(cx, e, op.node, lhs, rhs);
            },
            ExprKind::Assign(lhs, rhs, _) => {
                assign_op_pattern::check(cx, e, lhs, rhs);
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
