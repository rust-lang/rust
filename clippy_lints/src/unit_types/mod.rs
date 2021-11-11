mod let_unit_value;
mod unit_arg;
mod unit_cmp;
mod utils;

use rustc_hir::{Expr, Stmt};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for binding a unit value.
    ///
    /// ### Why is this bad?
    /// A unit value cannot usefully be used anywhere. So
    /// binding one is kind of pointless.
    ///
    /// ### Example
    /// ```rust
    /// let x = {
    ///     1;
    /// };
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub LET_UNIT_VALUE,
    pedantic,
    "creating a `let` binding to a value of unit type, which usually can't be used afterwards"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for comparisons to unit. This includes all binary
    /// comparisons (like `==` and `<`) and asserts.
    ///
    /// ### Why is this bad?
    /// Unit is always equal to itself, and thus is just a
    /// clumsily written constant. Mostly this happens when someone accidentally
    /// adds semicolons at the end of the operands.
    ///
    /// ### Example
    /// ```rust
    /// # fn foo() {};
    /// # fn bar() {};
    /// # fn baz() {};
    /// if {
    ///     foo();
    /// } == {
    ///     bar();
    /// } {
    ///     baz();
    /// }
    /// ```
    /// is equal to
    /// ```rust
    /// # fn foo() {};
    /// # fn bar() {};
    /// # fn baz() {};
    /// {
    ///     foo();
    ///     bar();
    ///     baz();
    /// }
    /// ```
    ///
    /// For asserts:
    /// ```rust
    /// # fn foo() {};
    /// # fn bar() {};
    /// assert_eq!({ foo(); }, { bar(); });
    /// ```
    /// will always succeed
    #[clippy::version = "pre 1.29.0"]
    pub UNIT_CMP,
    correctness,
    "comparing unit values"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for passing a unit value as an argument to a function without using a
    /// unit literal (`()`).
    ///
    /// ### Why is this bad?
    /// This is likely the result of an accidental semicolon.
    ///
    /// ### Example
    /// ```rust,ignore
    /// foo({
    ///     let a = bar();
    ///     baz(a);
    /// })
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub UNIT_ARG,
    complexity,
    "passing unit to a function"
}

declare_lint_pass!(UnitTypes => [LET_UNIT_VALUE, UNIT_CMP, UNIT_ARG]);

impl LateLintPass<'_> for UnitTypes {
    fn check_stmt(&mut self, cx: &LateContext<'_>, stmt: &Stmt<'_>) {
        let_unit_value::check(cx, stmt);
    }

    fn check_expr(&mut self, cx: &LateContext<'_>, expr: &Expr<'_>) {
        unit_cmp::check(cx, expr);
        unit_arg::check(cx, expr);
    }
}
