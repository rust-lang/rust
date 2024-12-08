//@ run-pass
//@ ignore-emscripten FIXME(#45351)

#![feature(repr_simd, intrinsics)]

#[repr(simd)]
#[derive(Copy, Clone, Debug)]
pub struct Char3(pub [i8; 3]);

#[repr(simd)]
#[derive(Copy, Clone, Debug)]
pub struct Short3(pub [i16; 3]);

extern "rust-intrinsic" {
    fn simd_cast<T, U>(x: T) -> U;
}

fn main() {
    let cast: Short3 = unsafe { simd_cast(Char3([10, -3, -9])) };

    println!("{:?}", cast);
}
