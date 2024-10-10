//@ compile-flags: -Znext-solver
#![allow(incomplete_features)]
#![feature(const_trait_impl, effects)]
#![feature(staged_api)]
#![stable(feature = "rust1", since = "1.0.0")]

#[stable(feature = "rust1", since = "1.0.0")]
#[const_trait]
pub trait MyTrait {
    #[stable(feature = "rust1", since = "1.0.0")]
    fn func();
}

#[stable(feature = "rust1", since = "1.0.0")]
pub struct Unstable;

#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_unstable(feature = "unstable", issue = "none")]
impl const MyTrait for Unstable {
    fn func() {}
}

#[stable(feature = "rust1", since = "1.0.0")]
pub struct Unstable2;

#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_unstable(feature = "unstable2", issue = "none")]
impl const MyTrait for Unstable2 {
    fn func() {}
}
