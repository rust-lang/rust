// run-pass
#![allow(dead_code)]

// pretty-expanded FIXME #23616

#![feature(repr_simd)]

#[repr(simd)]
struct RGBA {
    r: f32,
    g: f32,
    b: f32,
    a: f32
}

pub fn main() {}
