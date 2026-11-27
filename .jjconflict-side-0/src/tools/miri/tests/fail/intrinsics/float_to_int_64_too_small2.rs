#![feature(core_intrinsics)]
// Directly call intrinsic to avoid debug assertions in libstd
use std::intrinsics::float_to_int_unchecked;

fn main() {
    unsafe {
        float_to_int_unchecked::<f64, i64>(-9223372036854777856.0f64); //~ ERROR: cannot be represented in target type `i64`
    }
}
