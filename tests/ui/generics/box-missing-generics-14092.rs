//! Regression test for https://github.com/rust-lang/rust/issues/14092

fn fn1(0: Box) {}
//~^ ERROR missing generics for struct `Box`

fn main() {}
