//@ run-pass
#![allow(non_camel_case_types)]
#![feature(repr_simd, core_intrinsics)]

use std::intrinsics::simd::simd_add;
use std::ops;

#[repr(simd)]
#[derive(Copy, Clone)]
struct f32x4([f32; 4]);
impl f32x4 {
    fn to_array(self) -> [f32; 4] { unsafe { std::mem::transmute(self) } }
}

#[repr(simd)]
#[derive(Copy, Clone)]
struct A<const N: usize>([f32; N]);
impl<const N: usize> A<N> {
    fn to_array(self) -> [f32; N] { unsafe { std::intrinsics::transmute_unchecked(self) } }
}

#[repr(simd)]
#[derive(Copy, Clone)]
struct B<T>([T; 4]);
impl<T> B<T> {
    fn to_array(self) -> [T; 4] { unsafe { std::intrinsics::transmute_unchecked(self) } }
}

#[repr(simd)]
#[derive(Copy, Clone)]
struct C<T, const N: usize>([T; N]);
impl<T, const N: usize> C<T, N> {
    fn to_array(self) -> [T; N] { unsafe { std::intrinsics::transmute_unchecked(self) } }
}

fn add<T: ops::Add<Output = T>>(lhs: T, rhs: T) -> T {
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
    let a = f32x4([1.0f32, 2.0f32, 3.0f32, 4.0f32]);
    let [a0, a1, a2, a3] = add(a, a).to_array();
    assert_eq!(a0, 2.0f32);
    assert_eq!(a1, 4.0f32);
    assert_eq!(a2, 6.0f32);
    assert_eq!(a3, 8.0f32);

    let a = A(x);
    assert_eq!(add(a, a).to_array(), y);

    let b = B(x);
    assert_eq!(add(b, b).to_array(), y);

    let c = C(x);
    assert_eq!(add(c, c).to_array(), y);
}
