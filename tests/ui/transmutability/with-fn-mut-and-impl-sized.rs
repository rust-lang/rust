// Regression test for: https://github.com/rust-lang/rust/issues/153357
//@ revisions: old next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ [next] compile-flags: -Znext-solver

#![feature(transmutability)]
#![feature(unboxed_closures)]

use std::cell::Ref;

fn delay<Make: FnMut(Ref) -> C, C: std::mem::TransmuteFrom()>(make: Make) -> impl Sized {}
//~^ ERROR: missing generics for struct `Ref`
//~^^ ERROR: associated type `Output` not found for `TransmuteFrom`

fn main() {
    delay(|0| delay)
}
