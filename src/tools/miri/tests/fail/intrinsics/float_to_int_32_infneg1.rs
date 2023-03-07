#![feature(intrinsics)]

// Directly call intrinsic to avoid debug assertions in libstd
extern "rust-intrinsic" {
    fn float_to_int_unchecked<Float: Copy, Int: Copy>(value: Float) -> Int;
}

fn main() {
    unsafe {
        float_to_int_unchecked::<f32, i32>(f32::NEG_INFINITY); //~ ERROR: cannot be represented in target type `i32`
    }
}
