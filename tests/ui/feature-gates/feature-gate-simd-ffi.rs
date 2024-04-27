#![feature(repr_simd)]
#![allow(dead_code)]

#[repr(simd)]
#[derive(Copy, Clone)]
struct LocalSimd(u8, u8);

extern "C" {
    fn baz() -> LocalSimd; //~ ERROR use of SIMD type
    fn qux(x: LocalSimd); //~ ERROR use of SIMD type
}

fn main() {}
