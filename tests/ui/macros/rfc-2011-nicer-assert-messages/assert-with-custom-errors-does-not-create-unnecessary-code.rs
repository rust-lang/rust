// -Zpanic_abort_tests makes this test work on panic=abort targets and
// it's a no-op on panic=unwind targets
//@ compile-flags: --test -Zpanic_abort_tests
//@ run-pass

#![feature(core_intrinsics, generic_assert)]

#[should_panic(expected = "Custom user message")]
#[test]
fn test() {
  assert!(1 == 3, "Custom user message");
}

fn main() {
}
