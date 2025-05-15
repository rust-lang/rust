//@ compile-flags: -Copt-level=3
#![crate_type = "lib"]

#[no_mangle]
pub fn get_len() -> usize {
    // CHECK-LABEL: @get_len
    // CHECK-NEXT: start:
    // CHECK-NEXT: tail call void @__rust_no_alloc_shim_is_unstable()
    // CHECK-NEXT: ret i{{[0-9]+}} 3
    [1, 2, 3].iter().collect::<Vec<_>>().len()
}
