#![crate_type = "lib"]

//@ compile-flags: -O

// MIR inlining now optimizes this code.

// CHECK-LABEL: @unwrap_combinators
// CHECK: icmp
// CHECK-NEXT: icmp
// CHECK-NEXT: select i1
// CHECK-NEXT: ret i1
#[no_mangle]
pub fn unwrap_combinators(a: Option<i32>, b: i32) -> bool {
    a.map(|t| t >= b).unwrap_or(false)
}
