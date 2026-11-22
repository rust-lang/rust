#![deny(clippy::default_lint)]
#![allow(clippy::missing_clippy_version_attribute)]
#![feature(rustc_private)]
#![feature(default_field_values)]

#[macro_use]
extern crate rustc_middle;
#[macro_use]
extern crate rustc_lint;

declare_tool_lint! {
    pub clippy::TEST_LINT,
    Warn,
    "",
    @in_external_macro = Report

}

declare_tool_lint! {
//~^ default_lint
    pub clippy::TEST_LINT_DEFAULT,
    Warn,
    "default lint description",
    @in_external_macro = Report

}

declare_lint_pass!(Pass => [TEST_LINT]);
declare_lint_pass!(Pass2 => [TEST_LINT_DEFAULT]);

fn main() {}
