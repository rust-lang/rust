use rustc_lint::{EarlyLintPass, EarlyContext};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_ast::ast::*;

declare_clippy_lint! {
    /// **What it does:**
    /// Checks for function invocations of the form `primitive::from_str_radix(s, 10)`
    ///
    /// **Why is this bad?**
    /// This specific common use case can be rewritten as `s.parse::<primitive>()`
    /// (and in most cases, the turbofish can be removed), which reduces code length
    /// and complexity.
    ///
    /// **Known problems:** None.
    /// 
    /// **Example:**
    ///
    /// ```rust
    /// let input: &str = get_input();
    /// let num = u16::from_str_radix(input, 10)?;
    /// ```
    /// Use instead:
    /// ```rust
    /// let input: &str = get_input();
    /// let num: u16 = input.parse()?;
    /// ```
    pub FROM_STR_RADIX_10,
    style,
    "default lint description"
}

declare_lint_pass!(FromStrRadix10 => [FROM_STR_RADIX_10]);

impl EarlyLintPass for FromStrRadix10 {}
