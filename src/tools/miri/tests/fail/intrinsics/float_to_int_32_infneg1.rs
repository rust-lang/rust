#![feature(core_intrinsics)]
// Directly call intrinsic to avoid debug assertions in libstd
use std::intrinsics::float_to_int_unchecked;

fn main() {
    unsafe {
        float_to_int_unchecked::<f32, i32>(f32::NEG_INFINITY); //~ ERROR: cannot be represented in target type `i32`
    }
}
