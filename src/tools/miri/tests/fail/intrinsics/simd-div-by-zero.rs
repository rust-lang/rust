#![feature(core_intrinsics, repr_simd)]

use std::intrinsics::simd::simd_div;

#[repr(simd)]
#[allow(non_camel_case_types)]
struct i32x2([i32; 2]);

fn main() {
    unsafe {
        let x = i32x2([1, 1]);
        let y = i32x2([1, 0]);
        simd_div(x, y); //~ERROR: Undefined Behavior: dividing by zero
    }
}
