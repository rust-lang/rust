// build-fail

#![feature(repr_simd, platform_intrinsics)]

// error-pattern:monomorphising SIMD type `Simd<65536_usize>` of length greater than 32768

#[repr(simd)]
struct Simd<const N: usize>([f32; N]);

fn main() {
    let _ = Simd::<65536>([0.; 65536]);
}
