// build-fail

#![feature(repr_simd, platform_intrinsics)]

// error-pattern:monomorphising SIMD type `Simd<65537_usize>` of length greater than 65536

#[repr(simd)]
struct Simd<const N: usize>([f32; N]);

fn main() {
    let _ = Simd::<65537>([0.; 65537]);
}
