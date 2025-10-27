//@ compile-flags: --crate-type rlib
//@ check-pass
// Uses manual desugaring of EII internals:
// Tests whether it's okay to implement an unsafe EII with an unsafe implementation.
#![feature(eii)]
#![feature(decl_macro)]
#![feature(rustc_attrs)]
#![feature(eii_internals)]

#[eii_macro_for(bar, "unsafe")]
#[rustc_builtin_macro(eii_macro)]
macro foo() {

}

unsafe extern "Rust" {
    safe fn bar(x: u64) -> u64;
}

#[unsafe(foo)]
fn other(x: u64) -> u64 {
    x
}

fn main() {
    bar(0);
}
