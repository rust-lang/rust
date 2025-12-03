//@ check-pass
//@ revisions: e2015 e2018 e2021 e2024
//@ [e2015] edition: 2015
//@ [e2018] edition: 2018
//@ [e2021] edition: 2021
//@ [e2024] edition: 2024

// This demonstrates that, under the behavior introduced in PR
// #139493, that the `panic` macro is resolved even with
// `no_implicit_prelude` and without the macro being explicitly
// imported.  This happens due to a hack introduced in PR #62086 that
// causes macros in the standard library prelude to be resolved even
// when `no_implicit_prelude` is present.  The fact that PR #139493
// adds `panic` to the standard library prelude then triggers this
// behavior.
//
// In Rust 2015, this code was already accepted for a different reason
// (due to resolution from the macro prelude not being suppressed with
// `no_implicit_prelude`.)
//
// It may not be desirable to extend the scope of this hack to macros
// for which it did not already have effect.

#![no_implicit_prelude]
fn f() {
    panic!();
}

fn main() {}
