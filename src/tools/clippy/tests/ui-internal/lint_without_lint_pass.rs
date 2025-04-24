#![deny(clippy::lint_without_lint_pass)]
#![allow(clippy::missing_clippy_version_attribute)]
#![feature(rustc_private)]

#[macro_use]
extern crate rustc_middle;
#[macro_use]
extern crate rustc_session;
extern crate rustc_lint;
use rustc_lint::{LintPass, LintVec};

declare_tool_lint! {
//~^ lint_without_lint_pass
    pub clippy::TEST_LINT,
    Warn,
    "",
    report_in_external_macro: true
}

declare_tool_lint! {
    pub clippy::TEST_LINT_REGISTERED,
    Warn,
    "",
    report_in_external_macro: true
}

declare_tool_lint! {
    pub clippy::TEST_LINT_REGISTERED_ONLY_IMPL,
    Warn,
    "",
    report_in_external_macro: true
}

pub struct Pass;
impl LintPass for Pass {
    fn name(&self) -> &'static str {
        "TEST_LINT"
    }
    fn get_lints(&self) -> LintVec {
        vec![TEST_LINT]
    }
}

declare_lint_pass!(Pass2 => [TEST_LINT_REGISTERED]);

pub struct Pass3;
impl_lint_pass!(Pass3 => [TEST_LINT_REGISTERED_ONLY_IMPL]);

fn main() {}
