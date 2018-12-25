// compile-flags: -C no-prepopulate-passes -O

#![crate_type = "lib"]

// CHECK-LABEL: @zip_copy
#[no_mangle]
pub fn zip_copy(xs: &[u8], ys: &mut [u8]) {
// CHECK: memcpy
    for (x, y) in xs.iter().zip(ys) {
        *y = *x;
    }
}

// CHECK-LABEL: @zip_copy_mapped
#[no_mangle]
pub fn zip_copy_mapped(xs: &[u8], ys: &mut [u8]) {
// CHECK: memcpy
    for (x, y) in xs.iter().map(|&x| x).zip(ys) {
        *y = x;
    }
}
