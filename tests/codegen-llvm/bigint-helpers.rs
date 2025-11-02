//@ compile-flags: -C opt-level=3

#![crate_type = "lib"]
#![feature(bigint_helper_methods)]

// Note that there's also an assembly test for this, which is what checks for
// the `ADC` (Add with Carry) instruction on x86 now that the IR we emit uses
// the preferred instruction phrasing instead of the intrinsic.

// CHECK-LABEL: @u32_carrying_add
#[no_mangle]
pub fn u32_carrying_add(a: u32, b: u32, c: bool) -> (u32, bool) {
    // CHECK: %[[AB:.+]] = add i32 {{%a, %b|%b, %a}}
    // CHECK: %[[O1:.+]] = icmp ult i32 %[[AB]], %a
    // CHECK: %[[CEXT:.+]] = zext i1 %c to i32
    // CHECK: %[[ABC:.+]] = add i32 %[[AB]], %[[CEXT]]
    // CHECK: %[[O2:.+]] = icmp ult i32 %[[ABC]], %[[AB]]
    // CHECK: %[[O:.+]] = or disjoint i1 %[[O1]], %[[O2]]
    // CHECK: insertvalue {{.+}}, i32 %[[ABC]], 0
    // CHECK: insertvalue {{.+}}, i1 %[[O]], 1
    u32::carrying_add(a, b, c)
}
