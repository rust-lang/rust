//! Regression test for <https://github.com/rust-lang/rust/issues/46302>.
//! This used to emit incorrect type mismatch label pointing at return type.

fn foo() {
  let s = "abc";
  let u: &str = if true { s[..2] } else { s };
  //~^ ERROR mismatched types
}

fn main() {
    foo();
}
