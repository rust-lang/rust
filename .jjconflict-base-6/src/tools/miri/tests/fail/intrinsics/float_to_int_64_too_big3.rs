#![feature(core_intrinsics)]
// Directly call intrinsic to avoid debug assertions in libstd
use std::intrinsics::float_to_int_unchecked;

fn main() {
    unsafe {
        float_to_int_unchecked::<f64, u64>(18446744073709551616.0f64); //~ ERROR: cannot be represented in target type `u64`
    }
}
