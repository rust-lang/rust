//! Regression test for <https://github.com/rust-lang/rust/issues/134162>.
//!
//! <https://github.com/rust-lang/rust/pull/110877> introduced RHS type hints for when a ty doesn't
//! support a bin op. In the suggestion path, there was a `delay_bug`.
//! <https://github.com/rust-lang/rust/pull/121208> converted this `delay_bug` to `bug`, which did
//! not trigger any test failures as we did not have test coverage for this particular case. This
//! manifested in an ICE as reported in <https://github.com/rust-lang/rust/issues/134162>.

//@ revisions: e2018 e2021 e2024
//@[e2018] edition: 2018
//@[e2021] edition: 2021
//@[e2024] edition: 2024

fn main() {
    struct X;
    let _ = [X] == [panic!(); 2];
    //[e2018,e2021,e2024]~^ ERROR binary operation `==` cannot be applied to type `[X; 1]`
}
