//@ build-pass
//@ compile-flags: -Clink-dead-code
//@ needs-asm-support

#![allow(unused)]

// Test that a symbol in a `global_asm` namespace doesn't cause an ICE during v0 symbol mangling
// due to a lack of missing namespace character for `global_asm`.
//
// FIXME: Can't use `#[rustc_symbol_name]` on the `foo` call to check its symbol, so just checking
// the test compiles.

fn foo<const N: usize>() {}

core::arch::global_asm!("/* {} */", sym foo::<{
    || {};
    0
}>);

fn main() {}
