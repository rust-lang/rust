//@ ignore-backends: gcc
// FIXME(#125418): linking on Windows GNU targets is not yet supported.
//@ ignore-windows-gnu
// Tests whether one function could implement two EIIs.
#![feature(extern_item_impls)]

#[eii(a)]
static A: u64;

#[eii(b)]
static B: u64;

#[a]
#[b]
//~^ ERROR static cannot implement multiple EIIs
static IMPL: u64 = 5;

fn main() {
    println!("{A} {B} {IMPL}")
}
