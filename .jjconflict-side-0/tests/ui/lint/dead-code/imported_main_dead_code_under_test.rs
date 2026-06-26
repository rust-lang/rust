//@ check-pass
//@ compile-flags: --test

// Regression test for https://github.com/rust-lang/rust/issues/157608: a function used
// as `main` via a rename import was wrongly reported as dead code under `--test`.

#![deny(dead_code)]

fn different_main() {
    println!("Hello from different_main");
}

use different_main as main;
