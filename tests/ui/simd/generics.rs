// run-pass
#![allow(non_camel_case_types)]
#![feature(repr_simd, platform_intrinsics)]

use std::ops;

#[repr(simd)]
#[derive(Copy, Clone)]
struct f32x4(f32, f32, f32, f32);

#[repr(simd)]
#[derive(Copy, Clone)]
struct A<const N: usize>([f32; N]);

#[repr(simd)]
#[derive(Copy, Clone)]
struct B<T>([T; 4]);

#[repr(simd)]
#[derive(Copy, Clone)]
struct C<T, const N: usize>([T; N]);


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

impl ops::Add for A<4> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        unsafe { simd_add(self, rhs) }
    }
}

impl ops::Add for B<f32> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        unsafe { simd_add(self, rhs) }
    }
}

impl ops::Add for C<f32, 4> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        unsafe { simd_add(self, rhs) }
    }
}


pub fn main() {
    let x = [1.0f32, 2.0f32, 3.0f32, 4.0f32];
    let y = [2.0f32, 4.0f32, 6.0f32, 8.0f32];

    // lame-o
    let a = f32x4(1.0f32, 2.0f32, 3.0f32, 4.0f32);
    let f32x4(a0, a1, a2, a3) = add(a, a);
    assert_eq!(a0, 2.0f32);
    assert_eq!(a1, 4.0f32);
    assert_eq!(a2, 6.0f32);
    assert_eq!(a3, 8.0f32);

    let a = A(x);
    assert_eq!(add(a, a).0, y);

    let b = B(x);
    assert_eq!(add(b, b).0, y);

    let c = C(x);
    assert_eq!(add(c, c).0, y);
}
