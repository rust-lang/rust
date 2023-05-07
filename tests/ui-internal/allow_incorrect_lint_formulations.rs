#![allow(clippy::almost_standard_lint_formulation)]
#![feature(rustc_private)]

#[macro_use]
extern crate rustc_middle;
#[macro_use]
extern crate rustc_session;
extern crate rustc_lint;

declare_tool_lint! {
    /// # What it does
    /// Detects uses of incorrect formulations
    #[clippy::version = "pre 1.29.0"]
    pub clippy::ALLOWED_INVALID,
    Warn,
    "One",
    report_in_external_macro: true
}

declare_lint_pass!(Pass => [ALLOWED_INVALID]);

fn main() {}
