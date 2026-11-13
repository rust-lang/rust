//@ check-pass
//@aux-build:proc_macros.rs

#![warn(clippy::deref_addrof)]

extern crate proc_macros;

#[proc_macros::inline_macros]
fn f() -> i32 {
    // should be fine
    *inline!(&$1)
}

fn main() {}
