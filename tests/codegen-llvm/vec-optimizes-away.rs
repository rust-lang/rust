//@ compile-flags: -Copt-level=3
#![crate_type = "lib"]

#[no_mangle]
pub fn sum_me() -> i32 {
    // CHECK-LABEL: @sum_me
    // CHECK-NEXT: {{^.*:$}}
    // CHECK-NEXT: ; call __rustc::__rust_no_alloc_shim_is_unstable_v2
    // CHECK-NEXT: tail call void @_R{{.+}}__rust_no_alloc_shim_is_unstable_v2()
    // CHECK-NEXT: ret i32 6
    vec![1, 2, 3].iter().sum::<i32>()
}
