//@ compile-flags: -Copt-level=3

#![crate_type = "lib"]

// CHECK-LABEL: @zip_copy_mapped = unnamed_addr alias void (ptr, i64, ptr, i64), ptr @zip_copy

// CHECK-LABEL: @zip_copy
#[no_mangle]
pub fn zip_copy(xs: &[u8], ys: &mut [u8]) {
    // CHECK: memcpy
    for (x, y) in xs.iter().zip(ys) {
        *y = *x;
    }
}

#[no_mangle]
pub fn zip_copy_mapped(xs: &[u8], ys: &mut [u8]) {
    for (x, y) in xs.iter().map(|&x| x).zip(ys) {
        *y = x;
    }
}
