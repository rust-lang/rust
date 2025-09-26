//@ build-fail

#![feature(repr_simd, intrinsics)]

#[repr(simd)]
struct Simd<const N: usize>([f32; N]);

fn main() {
    let _x = Simd::<65536>([0.; 65536]);
    //~^ ERROR the SIMD type `Simd<65536>` has more elements than the limit 32768
}
