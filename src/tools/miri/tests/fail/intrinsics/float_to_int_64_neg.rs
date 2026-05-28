#![feature(core_intrinsics)]
// Directly call intrinsic to avoid debug assertions in libstd
use std::intrinsics::float_to_int_unchecked;

fn main() {
    unsafe {
        float_to_int_unchecked::<f64, u128>(-1.0000000000001f64); //~ ERROR: cannot be represented in target type `u128`
    }
}
