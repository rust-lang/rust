//@ ignore-debug: the debug assertions get in the way
//@ compile-flags: -O
#![crate_type = "lib"]

#[no_mangle]
pub fn sum_me() -> i32 {
    // CHECK-LABEL: @sum_me
    // CHECK-NEXT: {{^.*:$}}
    // CHECK-NEXT: {{.*}} load volatile i8, ptr @__rust_no_alloc_shim_is_unstable, align 1
    // CHECK-NEXT: ret i32 6
    vec![1, 2, 3].iter().sum::<i32>()
}
