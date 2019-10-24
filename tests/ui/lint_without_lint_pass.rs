#![deny(clippy::internal)]
#![feature(rustc_private)]

#[macro_use]
extern crate rustc;
use rustc::lint::{LintArray, LintPass};

#[macro_use]
extern crate clippy_lints;

declare_clippy_lint! {
    pub TEST_LINT,
    correctness,
    ""
}

declare_clippy_lint! {
    pub TEST_LINT_REGISTERED,
    correctness,
    ""
}

declare_clippy_lint! {
    pub TEST_LINT_REGISTERED_ONLY_IMPL,
    correctness,
    ""
}

pub struct Pass;
impl LintPass for Pass {
    fn name(&self) -> &'static str {
        "TEST_LINT"
    }
}

declare_lint_pass!(Pass2 => [TEST_LINT_REGISTERED]);

pub struct Pass3;
impl_lint_pass!(Pass3 => [TEST_LINT_REGISTERED_ONLY_IMPL]);

fn main() {}
