//@ build-pass
//@ aux-crate:simd=simd-lane-limit.rs

extern crate simd;

use simd::Simd;

fn main() {
    let _x: Simd<i32, 4> = Simd([0; 4]);
    let _y: Simd<i32, 8> = Simd([0; 8]);

    // test non-power-of-two, since #[repr(simd, packed)] has unusual layout
    let _z: Simd<i32, 6> = Simd([0; 6]);
}
