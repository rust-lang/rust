//! Ensure that `#[used]` in archives are correctly registered.
//!
//! Regression test for https://github.com/rust-lang/rust/issues/133491.

//@ run-pass
//@ check-run-results
//@ aux-build: used_pre_main_constructor.rs

// Make sure `rustc` links the archive, but intentionally do not import/use any items.
extern crate used_pre_main_constructor;

fn main() {
    // For some reason wasm-ld ignores the constructor unless we reference the
    // object file containing it.
    #[cfg(target_family = "wasm")]
    used_pre_main_constructor::force_constructor_call();

    println!("main");
}
