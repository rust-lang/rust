//@ build-fail
//@ aux-crate:simd=simd-lane-limit.rs

extern crate simd;

use simd::Simd;

fn main() {
    // test non-power-of-two, since #[repr(simd, packed)] has unusual layout
    let _x: Simd<i32, 24> = Simd([0; 24]);
    //~^ ERROR the SIMD type `simd::Simd<i32, 24>` has more elements than the limit 8
}
