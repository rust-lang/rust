// Tests that there's no bounds check for the inner loop after the assert.

//@ compile-flags: -Copt-level=3

#![crate_type = "lib"]

// CHECK-LABEL: @zero
#[no_mangle]
pub fn zero(d: &mut [Vec<i32>]) {
    // CHECK-NOT: panic_bounds_check
    let n = d.len();
    for i in 0..n {
        assert!(d[i].len() == n);
        for j in 0..n {
            d[i][j] = 0;
        }
    }
}
