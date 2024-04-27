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
// Missing attribute test
///////////////////////
declare_tool_lint! {
    #[clippy::version]
    pub clippy::MISSING_ATTRIBUTE_ONE,
    Warn,
    "Two",
    report_in_external_macro: true
}

declare_tool_lint! {
    pub clippy::MISSING_ATTRIBUTE_TWO,
    Warn,
    "Two",
    report_in_external_macro: true
}

#[allow(clippy::missing_clippy_version_attribute)]
mod internal_clippy_lints {
    declare_tool_lint! {
        pub clippy::ALLOW_MISSING_ATTRIBUTE_ONE,
        Warn,
        "Two",
        report_in_external_macro: true
    }
}

use crate::internal_clippy_lints::ALLOW_MISSING_ATTRIBUTE_ONE;
declare_lint_pass!(Pass2 => [VALID_ONE, VALID_TWO, VALID_THREE, INVALID_ONE, INVALID_TWO, MISSING_ATTRIBUTE_ONE, MISSING_ATTRIBUTE_TWO, ALLOW_MISSING_ATTRIBUTE_ONE]);

fn main() {}
