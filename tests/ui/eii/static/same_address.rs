//@ run-pass
//@ check-run-results
//@ ignore-backends: gcc
// FIXME: linking on windows (specifically mingw) not yet supported, see tracking issue #125418
//@ ignore-windows
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
