//@ compile-flags: -O
//@ min-llvm-version: 17

#![crate_type = "lib"]

#[no_mangle]
// CHECK-LABEL: @foo
// CHECK: getelementptr inbounds
// CHECK-NEXT: load i64
// CHECK-NEXT: icmp eq i64
// CHECK-NEXT: br i1
#[no_mangle]
pub fn foo(input: &mut &[u64]) -> Option<u64> {
    let (first, rest) = input.split_first()?;
    *input = rest;
    Some(*first)
}
