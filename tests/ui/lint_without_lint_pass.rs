#![deny(clippy::internal)]

#![feature(rustc_private)]

#[macro_use]
extern crate rustc;

#[macro_use]
extern crate clippy_lints;

declare_clippy_lint!
{
    pub TEST_LINT,
    correctness,
    ""
}

fn main() {
}
