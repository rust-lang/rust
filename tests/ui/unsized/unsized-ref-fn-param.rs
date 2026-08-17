//! Regression test for <https://github.com/rust-lang/rust/issues/38954>.
//! Test we emit helpful error message instead of silently failing.

fn _test(ref _p: str) {}
//~^ ERROR the size for values of type

fn main() { }
