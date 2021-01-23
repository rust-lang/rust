// ignore-emscripten FIXME(#45351)

#![feature(repr_simd, platform_intrinsics)]

#[repr(simd)]
#[derive(Copy, Clone, Debug)]
pub struct Char3(pub i8, pub i8, pub i8);
//~^ ERROR SIMD vector length must be a power of two

#[repr(simd)]
#[derive(Copy, Clone, Debug)]
pub struct Short3(pub i16, pub i16, pub i16);
//~^ ERROR SIMD vector length must be a power of two

extern "platform-intrinsic" {
    fn simd_cast<T, U>(x: T) -> U;
}

fn main() {
    let cast: Short3 = unsafe { simd_cast(Char3(10, -3, -9)) };

    println!("{:?}", cast);
}
