//@ compile-flags: -Copt-level=3
#![crate_type = "lib"]

#[no_mangle]
pub fn sum_me() -> i32 {
    // CHECK-LABEL: @sum_me
    // CHECK-NEXT: {{^.*:$}}
    // CHECK-NEXT: tail call void @__rust_no_alloc_shim_is_unstable()
    // CHECK-NEXT: ret i32 6
    vec![1, 2, 3].iter().sum::<i32>()
}
