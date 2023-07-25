//@compile-flags: --test
//@run

#![feature(core_intrinsics, generic_assert)]

#[should_panic(expected = "Custom user message")]
#[test]
fn test() {
  assert!(1 == 3, "Custom user message");
}

fn main() {
}
