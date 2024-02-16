//@ check-pass

// Tests that the unknown_lints lint doesn't fire for an unknown lint loaded from a separate file.
// The key part is that the stderr output should be empty.
// Reported in https://github.com/rust-lang/rust/issues/84936
// Fixed incidentally by https://github.com/rust-lang/rust/pull/97266

// This `allow` should apply to submodules, whether they are inline or loaded from a file.
#![allow(unknown_lints)]
#![allow(dead_code)]
// no warning
#![allow(not_a_real_lint)]

mod other;

// no warning
#[allow(not_a_real_lint)]
fn m() {}

mod mm {
    // no warning
    #[allow(not_a_real_lint)]
    fn m() {}
}

fn main() {}
