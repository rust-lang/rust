//@ compile-flags: -C opt-level=3

#![crate_type = "lib"]
#![feature(bigint_helper_methods)]

// CHECK-LABEL: @u32_carrying_add
#[no_mangle]
pub fn u32_carrying_add(a: u32, b: u32, c: bool) -> (u32, bool) {
    // CHECK: @llvm.uadd.with.overflow.i32
    // CHECK: @llvm.uadd.with.overflow.i32
    // CHECK: or disjoint i1
    u32::carrying_add(a, b, c)
}
