//@ compile-flags: -Copt-level=3
#![crate_type = "lib"]

#[no_mangle]
pub fn alloc_test(data: u32) {
    // CHECK-LABEL: @alloc_test
    // CHECK-NEXT: start:
    // CHECK-NEXT: tail call void @__rust_no_alloc_shim_is_unstable()
    let x = Box::new(data);
    drop(x);
}
