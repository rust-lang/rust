//@ run-rustfix
//@ rustfix-only-machine-applicable
//@ edition: 2024

// Verify that the suggestion produced by `impl_trait_redundant_captures`
// removes the adjacent `+` joiner along with `use<...>`, instead of leaving
// behind a stray `+` that fails to compile. Regression test for
// https://github.com/rust-lang/rust/issues/143216.

#![allow(unused)]
#![deny(impl_trait_redundant_captures)]

// `use<>` at the end of the bound list: the suggestion must remove the
// preceding `+`.
fn end_position() -> impl Sized + use<> {}
//~^ ERROR all possible in-scope parameters are already captured

// `use<>` at the start of the bound list: the suggestion must remove the
// following `+`.
fn start_position() -> impl use<> + Sized {}
//~^ ERROR all possible in-scope parameters are already captured

// `use<>` in the middle of the bound list: the suggestion must remove
// exactly one `+`, keeping the other to join the remaining bounds.
fn middle_position() -> impl Sized + use<> + Send {}
//~^ ERROR all possible in-scope parameters are already captured

fn main() {}
