// ignore-tidy-linelength

#![feature(const_generics)]

#![crate_name = "foo"]

use std::ops::Add;

// @has foo/struct.Simd.html '//pre[@class="rust struct"]' 'pub struct Simd<T, const WIDTH: usize>'
pub struct Simd<T, const WIDTH: usize> {
    inner: T,
}

// @has foo/struct.Simd.html '//div[@id="implementations-list"]/h3/code' 'impl Add<Simd<u8, 16>> for Simd<u8, 16>'
impl Add for Simd<u8, 16> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self { inner: 0 }
    }
}
