//@ check-pass
//! Tests that associated type projections normalize properly in the presence of HRTBs.
//! Original issue: <https://github.com/rust-lang/rust/issues/30472>


pub trait MyFrom<T> {}
impl<T> MyFrom<T> for T {}

pub trait MyInto<T> {}
impl<T, U> MyInto<U> for T where U: MyFrom<T> {}


pub trait A<'self_> {
    type T;
}
pub trait B: for<'self_> A<'self_> {
    // Originally caused the `type U = usize` example below to fail with a type mismatch error
    type U: for<'self_> MyFrom<<Self as A<'self_>>::T>;
}


pub struct M;
impl<'self_> A<'self_> for M {
    type T = usize;
}

impl B for M {
    type U = usize;
}


fn main() {}
