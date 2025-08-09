mod let_unit_value;
mod unit_arg;
mod unit_cmp;
mod utils;

use clippy_utils::macros::FormatArgsStorage;
use rustc_hir::{Expr, LetStmt};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::impl_lint_pass;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for binding a unit value.
    ///
    /// ### Why is this bad?
    /// A unit value cannot usefully be used anywhere. So
    /// binding one is kind of pointless.
    ///
    /// ### Example
    /// ```no_run
    /// let x = {
    ///     1;
    /// };
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub LET_UNIT_VALUE,
    style,
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
    /// ```no_run
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
    /// ```no_run
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
    /// ```no_run
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

pub struct UnitTypes {
    format_args: FormatArgsStorage,
}

impl_lint_pass!(UnitTypes => [LET_UNIT_VALUE, UNIT_CMP, UNIT_ARG]);

impl UnitTypes {
    pub fn new(format_args: FormatArgsStorage) -> Self {
        Self { format_args }
    }
}

impl<'tcx> LateLintPass<'tcx> for UnitTypes {
    fn check_local(&mut self, cx: &LateContext<'tcx>, local: &'tcx LetStmt<'tcx>) {
        let_unit_value::check(cx, &self.format_args, local);
    }

    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) {
        unit_cmp::check(cx, expr);
        unit_arg::check(cx, expr);
    }
}
