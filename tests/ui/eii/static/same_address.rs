//@ run-pass
//@ check-run-results
// FIXME(#125418): linking on Windows GNU targets is not yet supported.
//@ ignore-windows-gnu
// FIXME GCC seems to treat integers as if they have provenance, causing the comparison to fail
// despite identical address.
//@ ignore-backends: gcc
// Tests whether EIIs and their declarations share the same address
#![feature(extern_item_impls)]

#[eii(hello)]
static HELLO: u64;

#[hello]
static HELLO_IMPL: u64 = 5;

// what you would write:
fn main() {
    assert_eq!(
        &HELLO as *const u64 as usize,
        &HELLO_IMPL as *const u64 as usize,
    )
}
