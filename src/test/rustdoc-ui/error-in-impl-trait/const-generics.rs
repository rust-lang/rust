// check-pass
// edition:2018
#![feature(min_const_generics)]
trait ValidTrait {}

/// This has docs
pub fn extern_fn<const N: usize>() -> impl Iterator<Item = [u8; N]> {
    loop {}
}

pub trait Trait<const N: usize> {}
impl Trait<1> for u8 {}
impl Trait<2> for u8 {}
impl<const N: usize> Trait<N> for [u8; N] {}

/// This also has docs
pub fn test<const N: usize>() -> impl Trait<N> where u8: Trait<N> {
    loop {}
}

/// Document all the functions
pub async fn a_sink<const N: usize>(v: [u8; N]) -> impl Trait<N> {
    loop {}
}
