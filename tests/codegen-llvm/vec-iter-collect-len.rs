//@ compile-flags: -Copt-level=3
#![crate_type = "lib"]

#[no_mangle]
pub fn get_len() -> usize {
    // CHECK-LABEL: @get_len
    // CHECK-NEXT: start:
    // CHECK-NEXT: ; call __rustc::__rust_no_alloc_shim_is_unstable_v2
    // CHECK-NEXT: tail call void @_R{{.+}}__rust_no_alloc_shim_is_unstable_v2()
    // CHECK-NEXT: ret i{{[0-9]+}} 3
    [1, 2, 3].iter().collect::<Vec<_>>().len()
}
