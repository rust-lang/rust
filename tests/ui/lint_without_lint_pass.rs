#![deny(clippy::internal)]
#![feature(rustc_private)]

#[macro_use]
extern crate rustc;
use rustc::lint;

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

pub struct Pass;
impl lint::LintPass for Pass {
    fn get_lints(&self) -> lint::LintArray {
        lint_array!(TEST_LINT_REGISTERED)
    }

    fn name(&self) -> &'static str {
        "TEST_LINT"
    }
}

fn main() {}
