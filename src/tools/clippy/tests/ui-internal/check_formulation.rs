#![deny(clippy::almost_standard_lint_formulation)]
#![allow(clippy::lint_without_lint_pass)]
#![feature(rustc_private)]

#[macro_use]
extern crate rustc_middle;
#[macro_use]
extern crate rustc_session;
extern crate rustc_lint;

declare_tool_lint! {
    /// # What it does
    ///
    /// Checks for usage of correct lint formulations
    #[clippy::version = "pre 1.29.0"]
    pub clippy::VALID,
    Warn,
    "One",
    report_in_external_macro: true
}

declare_tool_lint! {
    /// # What it does
    /// Check for lint formulations that are correct
    //~^ almost_standard_lint_formulation
    #[clippy::version = "pre 1.29.0"]
    pub clippy::INVALID1,
    Warn,
    "One",
    report_in_external_macro: true
}

declare_tool_lint! {
    /// # What it does
    /// Detects uses of incorrect formulations
    //~^ almost_standard_lint_formulation
    #[clippy::version = "pre 1.29.0"]
    pub clippy::INVALID2,
    Warn,
    "One",
    report_in_external_macro: true
}

declare_tool_lint! {
    /// # What it does
    /// Detects uses of incorrect formulations (allowed with attribute)
    #[allow(clippy::almost_standard_lint_formulation)]
    #[clippy::version = "pre 1.29.0"]
    pub clippy::ALLOWED_INVALID,
    Warn,
    "One",
    report_in_external_macro: true
}

declare_lint_pass!(Pass => [VALID, INVALID1, INVALID2]);

fn main() {}
