// aux-build:common.rs
// run-pass

#![feature(core_intrinsics, generic_assert, generic_assert_internals)]

extern crate common;

fn main() {
  common::test!(
    let mut _nothing = ();
    [ 1 == 3 ] => "Assertion failed: 1 == 3"
  );
}
