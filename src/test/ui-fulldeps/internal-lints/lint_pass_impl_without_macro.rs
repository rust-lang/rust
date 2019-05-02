// compile-flags: -Z unstable-options

#![feature(rustc_private)]
#![deny(lint_pass_impl_without_macro)]

extern crate rustc;

use rustc::lint::{LintArray, LintPass};
use rustc::{declare_lint, declare_lint_pass, impl_lint_pass, lint_array};

declare_lint! {
    pub TEST_LINT,
    Allow,
    "test"
}

struct Foo;

impl LintPass for Foo { //~ERROR implementing `LintPass` by hand
    fn get_lints(&self) -> LintArray {
        lint_array!(TEST_LINT)
    }

    fn name(&self) -> &'static str {
        "Foo"
    }
}

struct Bar;

impl_lint_pass!(Bar => [TEST_LINT]);

declare_lint_pass!(Baz => [TEST_LINT]);

fn main() {}
