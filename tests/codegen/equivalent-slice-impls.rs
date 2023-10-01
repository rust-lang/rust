// This test checks that two equivalent implementations produce the same number of `if` instructions.

#![crate_type = "lib"]

// CHECK-LABEL: @f1
// CHECK: if
pub fn f1(input: &mut &[u64]) -> Option<u64> {
  match input {
      [] => None,
      [first, rest @ ..] => {
          *input = rest;
          Some(*first)
      }
  }
}

// CHECK-LABEL: @f2
// CHECK: if
pub fn f2(input: &mut &[u64]) -> Option<u64> {
  let (first, rest) = input.split_first()?;
  *input = rest;
  Some(*first)
}
