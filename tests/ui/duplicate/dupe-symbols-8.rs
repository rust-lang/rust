//@ build-fail
//@ error-pattern: entry symbol `main` declared multiple times
//@ ignore-wasi wasi does different things with the `main` symbol
//
// See #67946.

#![allow(warnings)]
fn main() {
    extern "Rust" {
     fn main();
    }
    unsafe { main(); }
}
