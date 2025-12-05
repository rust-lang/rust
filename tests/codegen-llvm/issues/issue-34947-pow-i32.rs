//@ compile-flags: -Copt-level=3

#![crate_type = "lib"]

// CHECK-LABEL: @issue_34947
#[no_mangle]
pub fn issue_34947(x: i32) -> i32 {
    // CHECK: mul
    // CHECK-NEXT: mul
    // CHECK-NEXT: mul
    // CHECK-NEXT: ret
    x.pow(5)
}
