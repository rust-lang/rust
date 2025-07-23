#![feature(autodiff)]

use std::autodiff::autodiff_reverse;
use std::ptr;

#[autodiff_reverse(d_test_memcpy, Duplicated, Active)]
#[no_mangle]
fn test_memcpy(input: &[f64; 8]) -> f64 {
    let mut local_data = [0.0f64; 8];
    
    unsafe {
        ptr::copy_nonoverlapping(input.as_ptr(), local_data.as_mut_ptr(), 8);
    }
    
    let mut result = 0.0;
    for i in 0..8 {
        result += local_data[i] * local_data[i];
    }
    
    result
}

fn main() {
    let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mut d_input = [0.0; 8];
    let result = test_memcpy(&input);
    let result_d = d_test_memcpy(&input, &mut d_input, 1.0);
    
    assert_eq!(result, result_d);
    println!("Memcpy test passed: result = {}", result);
} 