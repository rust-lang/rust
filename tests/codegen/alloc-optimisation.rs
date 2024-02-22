//@ compile-flags: -O
#![crate_type = "lib"]

#[no_mangle]
pub fn alloc_test(data: u32) {
    // CHECK-LABEL: @alloc_test
    // CHECK-NEXT: start:
    // CHECK-NEXT: {{.*}} load volatile i8, ptr @__rust_no_alloc_shim_is_unstable, align 1
    // CHECK-NEXT: ret void
    let x = Box::new(data);
    drop(x);
}
