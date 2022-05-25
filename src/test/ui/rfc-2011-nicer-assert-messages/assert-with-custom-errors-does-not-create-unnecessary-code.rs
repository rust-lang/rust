// compile-flags: --test
// run-pass

#![feature(core_intrinsics, generic_assert, generic_assert_internals)]

#[should_panic(expected = "OMG!")]
#[test]
fn test() {
  assert!(1 == 3, "OMG!");
}

fn main() {
}
