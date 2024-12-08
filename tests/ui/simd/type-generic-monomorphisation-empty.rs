//@ build-fail

#![feature(repr_simd, intrinsics)]

//@ error-pattern:monomorphising SIMD type `Simd<0>` of zero length

#[repr(simd)]
struct Simd<const N: usize>([f32; N]);

fn main() {
    let _ = Simd::<0>([]);
}
