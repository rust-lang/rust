//@ build-fail
//@ aux-crate:simd=simd-lane-limit.rs

extern crate simd;

use simd::Simd;

fn main() {
    let _x: Simd<i32, 16> = Simd([0; 16]);
    //~^ ERROR the SIMD type `simd::Simd<i32, 16>` has more elements than the limit 8
}
