// build-fail

#![feature(repr_simd, platform_intrinsics)]

// error-pattern:monomorphising SIMD type `Simd<3_usize>` of non-power-of-two length

#[repr(simd)]
struct Simd<const N: usize>([f32; N]);

fn main() {
    let _ = Simd::<3>([0.; 3]);
}
