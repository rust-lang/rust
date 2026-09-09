//@ known-bug: #152204
//@ compile-flags: -Copt-level=0
#![feature(portable_simd)]

fn main() {
    if false {
        let _ = core::simd::Simd::<u8, 256>::splat(0);
    }
}
