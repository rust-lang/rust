//@ compile-flags: -Copt-level=3

#![crate_type = "lib"]

// CHECK-LABEL: @foo
// CHECK-NEXT: {{.*}}:
// CHECK-NEXT: getelementptr inbounds
// CHECK-NEXT: load [[TYPE:i(32|64)]]
// CHECK-NEXT: icmp eq [[TYPE]]
// CHECK-NEXT: br i1
#[no_mangle]
pub fn foo(input: &mut &[u64]) -> Option<u64> {
    let (first, rest) = input.split_first()?;
    *input = rest;
    Some(*first)
}
