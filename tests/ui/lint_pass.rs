
#![feature(rustc_private)]
#![feature(macro_vis_matcher)]

#![warn(lint_without_lint_pass)]

#[macro_use] extern crate rustc;

use rustc::lint::{LintPass, LintArray};

declare_clippy_lint! { GOOD_LINT, style, "good lint" }
declare_clippy_lint! { MISSING_LINT, style, "missing lint" }

pub struct Pass;

impl LintPass for Pass {
    fn get_lints(&self) -> LintArray {
        lint_array![GOOD_LINT]
    }
}

fn main() {
    let _ = MISSING_LINT;
}
