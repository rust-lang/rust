//@ build-fail
//@ ignore-wasi wasi does different things with the `main` symbol

#![allow(warnings)]

#[no_mangle]
fn main(){} //~ ERROR entry symbol `main` declared multiple times
