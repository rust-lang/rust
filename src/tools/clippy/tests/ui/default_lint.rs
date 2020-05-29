#![deny(clippy::internal)]
#![feature(rustc_private)]

#[macro_use]
extern crate rustc_middle;
#[macro_use]
extern crate rustc_session;
extern crate rustc_lint;

declare_tool_lint! {
    pub clippy::TEST_LINT,
    Warn,
    "",
    report_in_external_macro: true
}

declare_tool_lint! {
    pub clippy::TEST_LINT_DEFAULT,
    Warn,
    "default lint description",
    report_in_external_macro: true
}

declare_lint_pass!(Pass => [TEST_LINT]);
declare_lint_pass!(Pass2 => [TEST_LINT_DEFAULT]);

fn main() {}
