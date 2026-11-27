//@ compile-flags: -Znext-solver=globally

// Regression test for <https://github.com/rust-lang/rust/issues/151304>.
//
// This used to ICE with `-Znext-solver=globally`.
// The ICE happened because the `PartialOrd` bound fails causing
// diagnostics to replay the proof tree in order to find the
// best nested-goal. During that replay, it needs to create a
// fresh inference variable for the higher-ranked `T` but it was
// creating it in the wrong universe.

#![allow(incomplete_features)]
#![feature(non_lifetime_binders)]

fn auto_trait()
where
    for<T> T: PartialEq + PartialOrd,
{}

fn main() {
    auto_trait();
    //~^ ERROR can't compare `T` with `T`
}
