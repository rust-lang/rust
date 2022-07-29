#![crate_name = "foo"]

use std::ops::Add;

// @has foo/struct.Simd.html '//pre[@class="rust struct"]' 'pub struct Simd<T, const WIDTH: usize>'
pub struct Simd<T, const WIDTH: usize> {
    inner: T,
}

// @has foo/struct.Simd.html '//div[@id="trait-implementations-list"]//h3[@class="code-header in-band"]' 'impl Add<Simd<u8, 16_usize>> for Simd<u8, 16>'
impl Add for Simd<u8, 16> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self { inner: 0 }
    }
}
