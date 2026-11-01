#![feature(autodiff)]

use std::autodiff::autodiff_reverse;
use std::ptr;

#[inline(never)]
fn force_memcpy(src: *const f64, dst: *mut f64, count: usize) {
    unsafe {
        ptr::copy_nonoverlapping(src, dst, count);
    }
}

#[autodiff_reverse(d_test_memcpy, Duplicated, Active)]
#[no_mangle]
fn test_memcpy(input: &[f64; 128]) -> f64 {
    let mut local_data = [0.0f64; 128];

    // Use a separate function to prevent inlining and optimization
    force_memcpy(input.as_ptr(), local_data.as_mut_ptr(), 128);

    // Sum only first few elements to keep the computation simple
    local_data[0] * local_data[0]
        + local_data[1] * local_data[1]
        + local_data[2] * local_data[2]
        + local_data[3] * local_data[3]
}

fn main() {
    let input = [1.0; 128];
    let mut d_input = [0.0; 128];
    let result = test_memcpy(&input);
    let result_d = d_test_memcpy(&input, &mut d_input, 1.0);

    assert_eq!(result, result_d);
    println!("Memcpy test passed: result = {}", result);
}
