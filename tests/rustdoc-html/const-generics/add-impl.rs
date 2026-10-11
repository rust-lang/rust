#![crate_name = "foo"]

use std::ops::Add;

//@ has foo/type.Simd.html '//pre[@class="rust item-decl"]' 'pub struct Simd<T, const WIDTH: usize>'
pub struct Simd<T, const WIDTH: usize> {
    inner: T,
}

//@ has foo/type.Simd.html '//div[@id="trait-implementations-list"]//h3[@class="code-header"]' 'impl Add for Simd<u8, 16>'
impl Add for Simd<u8, 16> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self { inner: 0 }
    }
}
