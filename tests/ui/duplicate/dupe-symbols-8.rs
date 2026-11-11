//@ build-fail
//@ ignore-wasi wasi does different things with the `main` symbol
//
// See #67946.

#![allow(warnings)]
fn main() { //~ ERROR entry symbol `main` declared multiple times
    extern "Rust" {
     fn main();
    }
    unsafe { main(); }
}
