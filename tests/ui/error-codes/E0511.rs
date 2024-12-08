//@ build-fail

#![feature(intrinsics)]

extern "rust-intrinsic" {
    fn simd_add<T>(a: T, b: T) -> T;
}

fn main() {
    unsafe { simd_add(0, 1); } //~ ERROR E0511
}
