// run-pass
#![allow(non_snake_case)]

// ignore-emscripten FIXME(#45351)

#![feature(repr_simd, platform_intrinsics)]

#[repr(C)]
#[repr(simd)]
#[derive(Copy, Clone, Debug)]
pub struct char3(pub i8, pub i8, pub i8);

#[repr(C)]
#[repr(simd)]
#[derive(Copy, Clone, Debug)]
pub struct short3(pub i16, pub i16, pub i16);

extern "platform-intrinsic" {
    fn simd_cast<T, U>(x: T) -> U;
}

fn main() {
    let cast: short3 = unsafe { simd_cast(char3(10, -3, -9)) };

    println!("{:?}", cast);
}
