//! Regression test for https://github.com/rust-lang/rust/issues/10764

fn f(_: extern "Rust" fn()) {}
extern "C" fn bar() {}

fn main() { f(bar) }
//~^ ERROR mismatched types
