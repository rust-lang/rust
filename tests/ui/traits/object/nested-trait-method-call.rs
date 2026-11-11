//! Regression test for <https://github.com/rust-lang/rust/issues/3702>.
//! Calling method of trait defined in function used to trigger LLVM assertion.
//@ run-pass

#![allow(dead_code)]

pub fn main() {
  trait Text {
    fn to_string(&self) -> String;
  }

  fn to_string(t: Box<dyn Text>) {
    println!("{}", (*t).to_string());
  }

}
