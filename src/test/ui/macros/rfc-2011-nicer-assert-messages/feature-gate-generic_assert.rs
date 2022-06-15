// compile-flags: --test
// run-pass

// `generic_assert` is completely unimplemented and doesn't generate any logic, thus the
// reason why this test currently passes
#![feature(core_intrinsics, generic_assert, generic_assert_internals)]

use std::fmt::{Debug, Formatter};

#[derive(Clone, Copy, PartialEq)]
struct CopyDebug(i32);

impl Debug for CopyDebug {
  fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
    f.write_str("With great power comes great electricity bills")
  }
}

#[test]
fn test() {
  let _copy_debug = CopyDebug(1);
  assert!(_copy_debug == CopyDebug(3));
}

fn main() {
}
