//! Regression test for https://github.com/rust-lang/rust/issues/11820

//@ run-pass

#![allow(noop_method_call)]

struct NoClone;

fn main() {
  let rnc = &NoClone;
  let rsnc = &Some(NoClone);

  let _: &NoClone = rnc.clone();
  let _: &Option<NoClone> = rsnc.clone();
}
