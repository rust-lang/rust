//@ compile-flags: -Copt-level=3
#![crate_type = "lib"]

// CHECK-LABEL: @div2
// CHECK: ashr i32 %a, 1
// CHECK-NEXT: ret i32
#[no_mangle]
pub fn div2(a: i32) -> i32 {
    a.div_euclid(2)
}
