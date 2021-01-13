// run-pass
#![allow(non_camel_case_types)]
#![feature(repr_simd, platform_intrinsics)]

use std::ops;

#[repr(simd)]
#[derive(Copy, Clone)]
struct f32x4(f32, f32, f32, f32);

#[repr(simd)]
#[derive(Copy, Clone)]
struct S<const N: usize>([f32; N]);


extern "platform-intrinsic" {
    fn simd_add<T>(x: T, y: T) -> T;
}

fn add<T: ops::Add<Output=T>>(lhs: T, rhs: T) -> T {
    lhs + rhs
}

impl ops::Add for f32x4 {
    type Output = f32x4;

    fn add(self, rhs: f32x4) -> f32x4 {
        unsafe { simd_add(self, rhs) }
    }
}

impl ops::Add for S<4> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        unsafe { simd_add(self, rhs) }
    }
}


pub fn main() {
    let lr = f32x4(1.0f32, 2.0f32, 3.0f32, 4.0f32);

    // lame-o
    let f32x4(x, y, z, w) = add(lr, lr);
    assert_eq!(x, 2.0f32);
    assert_eq!(y, 4.0f32);
    assert_eq!(z, 6.0f32);
    assert_eq!(w, 8.0f32);

    let lr2 = S::<4>([1.0f32, 2.0f32, 3.0f32, 4.0f32]);
    let [x, y, z, w] = add(lr2, lr2).0;
    assert_eq!(x, 2.0f32);
    assert_eq!(y, 4.0f32);
    assert_eq!(z, 6.0f32);
    assert_eq!(w, 8.0f32);
}
