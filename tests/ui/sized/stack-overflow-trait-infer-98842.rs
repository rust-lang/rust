// #98842 stack overflow in trait inference
// issue: rust-lang/rust#98842
//@ check-fail
//@ edition:2021
//@ stderr-per-bitwidth
//~^^^^^ ERROR cycle detected when computing layout of `Foo`

// If the inner `Foo` is named through an associated type,
// the "infinite size" error does not occur.
struct Foo(<&'static Foo as ::core::ops::Deref>::Target);
// But Rust will be unable to know whether `Foo` is sized or not,
// and it will infinitely recurse somewhere trying to figure out the
// size of this pointer (is my guess):
const _: *const Foo = 0 as _;
//~^ ERROR a cycle occurred during layout computation

pub fn main() {}
