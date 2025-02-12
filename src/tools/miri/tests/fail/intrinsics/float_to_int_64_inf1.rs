#![feature(intrinsics)]

// Directly call intrinsic to avoid debug assertions in libstd
#[rustc_intrinsic]
unsafe fn float_to_int_unchecked<Float: Copy, Int: Copy>(_value: Float) -> Int;

fn main() {
    unsafe {
        float_to_int_unchecked::<f64, u128>(f64::INFINITY); //~ ERROR: cannot be represented in target type `u128`
    }
}
