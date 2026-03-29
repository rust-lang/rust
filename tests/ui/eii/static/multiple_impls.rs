//@ run-pass
//@ check-run-results
//@ ignore-backends: gcc
// FIXME: linking on windows (speciifcally mingw) not yet supported, see tracking issue #125418
//@ ignore-windows
// Tests whether one function could implement two EIIs.
#![feature(extern_item_impls)]

#[eii(a)]
static A: u64;

#[eii(b)]
static B: u64;

#[a]
#[b]
static IMPL: u64 = 5;

fn main() {
    println!("{A} {B} {IMPL}")
}
