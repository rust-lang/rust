#![feature(intrinsics)]

// Directly call intrinsic to avoid debug assertions in libstd
#[rustc_intrinsic]
unsafe fn float_to_int_unchecked<Float: Copy, Int: Copy>(value: Float) -> Int;

fn main() {
    unsafe {
        float_to_int_unchecked::<f32, u32>(-1.000000001f32); //~ ERROR: cannot be represented in target type `u32`
    }
}
