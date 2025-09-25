//@ compile-flags: -Copt-level=3

#![crate_type = "lib"]

// CHECK-LABEL: @issue_131162
#[no_mangle]
pub fn issue_131162(a1: usize, a2: usize) -> bool {
    const MASK: usize = 1;

    // CHECK-NOT: xor
    // CHECK-NOT: trunc
    // CHECK-NOT: and i1
    // CHECK: icmp
    // CHECK-NEXT: ret
    (a1 & !MASK) == (a2 & !MASK) && (a1 & MASK) == (a2 & MASK)
}
