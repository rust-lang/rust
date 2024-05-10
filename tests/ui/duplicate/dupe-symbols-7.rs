//@ build-fail
//@ ignore-wasi wasi does different things with the `main` symbol

//
//@ error-pattern: entry symbol `main` declared multiple times

#![allow(warnings)]

#[no_mangle]
fn main(){}
