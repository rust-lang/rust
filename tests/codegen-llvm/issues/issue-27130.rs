//@ compile-flags: -Copt-level=3

#![crate_type = "lib"]

// CHECK-LABEL: @trim_in_place
#[no_mangle]
pub fn trim_in_place(a: &mut &[u8]) {
    while a.first() == Some(&42) {
        // CHECK-NOT: slice_index_fail
        *a = &a[1..];
    }
}

// CHECK-LABEL: @trim_in_place2
#[no_mangle]
pub fn trim_in_place2(a: &mut &[u8]) {
    while let Some(&42) = a.first() {
        // CHECK-COUNT-1: slice_index_fail
        *a = &a[2..];
    }
}
