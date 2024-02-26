//@ build-fail

#![feature(repr_simd, intrinsics)]

//@ error-pattern:monomorphising SIMD type `Simd<65536>` of length greater than 32768

#[repr(simd)]
struct Simd<const N: usize>([f32; N]);

fn main() {
    let _ = Simd::<65536>([0.; 65536]);
}
