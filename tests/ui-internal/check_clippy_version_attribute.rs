#![deny(clippy::internal)]
#![feature(rustc_private)]

#[macro_use]
extern crate rustc_middle;
#[macro_use]
extern crate rustc_session;
extern crate rustc_lint;

///////////////////////
// Valid descriptions
///////////////////////
declare_tool_lint! {
    #[clippy::version = "pre 1.29.0"]
    pub clippy::VALID_ONE,
    Warn,
    "One",
    report_in_external_macro: true
}

declare_tool_lint! {
    #[clippy::version = "1.29.0"]
    pub clippy::VALID_TWO,
    Warn,
    "Two",
    report_in_external_macro: true
}

declare_tool_lint! {
    #[clippy::version = "1.59.0"]
    pub clippy::VALID_THREE,
    Warn,
    "Three",
    report_in_external_macro: true
}

///////////////////////
// Invalid attributes
///////////////////////
declare_tool_lint! {
    #[clippy::version = "1.2.3.4.5.6"]
    pub clippy::INVALID_ONE,
    Warn,
    "One",
    report_in_external_macro: true
}

declare_tool_lint! {
    #[clippy::version = "I'm a string"]
    pub clippy::INVALID_TWO,
    Warn,
    "Two",
    report_in_external_macro: true
}

///////////////////////
// Ignored attributes
///////////////////////
declare_tool_lint! {
    #[clippy::version]
    pub clippy::IGNORED_ONE,
    Warn,
    "ONE",
    report_in_external_macro: true
}

declare_lint_pass!(Pass2 => [VALID_ONE, VALID_TWO, VALID_THREE, INVALID_ONE, INVALID_TWO, IGNORED_ONE]);

fn main() {}
