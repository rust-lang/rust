//@ run-pass
#![allow(non_camel_case_types)]
#![feature(repr_simd, core_intrinsics)]

#[path = "../../auxiliary/minisimd.rs"]
mod minisimd;
use minisimd::*;

use std::intrinsics::simd::simd_add;
use std::ops;

type A<const N: usize> = Simd<f32, N>;

type B<T> = Simd<T, 4>;

type C<T, const N: usize> = Simd<T, N>;

fn add<T: ops::Add<Output = T>>(lhs: T, rhs: T) -> T {
    lhs + rhs
}

impl ops::Add for f32x4 {
    type Output = f32x4;

    fn add(self, rhs: f32x4) -> f32x4 {
        unsafe { simd_add(self, rhs) }
    }
}

pub fn main() {
    let x = [1.0f32, 2.0f32, 3.0f32, 4.0f32];
    let y = [2.0f32, 4.0f32, 6.0f32, 8.0f32];

    // lame-o
    let a = f32x4::from_array([1.0f32, 2.0f32, 3.0f32, 4.0f32]);
    let [a0, a1, a2, a3] = add(a, a).into_array();
    assert_eq!(a0, 2.0f32);
    assert_eq!(a1, 4.0f32);
    assert_eq!(a2, 6.0f32);
    assert_eq!(a3, 8.0f32);

    let a = A::from_array(x);
    assert_eq!(add(a, a).into_array(), y);

    let b = B::from_array(x);
    assert_eq!(add(b, b).into_array(), y);

    let c = C::from_array(x);
    assert_eq!(add(c, c).into_array(), y);
}
