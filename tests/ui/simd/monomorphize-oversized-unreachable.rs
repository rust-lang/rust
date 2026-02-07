// Regression test for #152204.
//@ build-fail
//@ only-x86_64

#![feature(portable_simd)]

fn main() {
    if false {
        let _ = core::simd::Simd::<u8, 256>::splat(0);
    }
}

//~? ERROR the SIMD type `Simd<u8, 256>` has more elements than the limit 64
