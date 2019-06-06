// compile-flags: -O

#![crate_type = "lib"]

// CHECK-LABEL: @issue_34947
#[no_mangle]
pub fn issue_34947(x: i32) -> i32 {
    // CHECK: mul
    // CHECK-NOT: br label
    // CHECK: mul
    // CHECK-NOT: br label
    // CHECK: mul
    // CHECK-NOT: br label
    // CHECK: ret
    x.pow(5)
}
