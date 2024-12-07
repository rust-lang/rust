//@ compile-flags: -Znext-solver
#![feature(const_trait_impl)]
#![feature(staged_api)]
#![stable(feature = "rust1", since = "1.0.0")]

#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_unstable(feature = "unstable", issue = "none")]
#[const_trait]
pub trait MyTrait {
    #[stable(feature = "rust1", since = "1.0.0")]
    fn func();
}

#[stable(feature = "rust1", since = "1.0.0")]
pub struct Unstable;

#[stable(feature = "rust1", since = "1.0.0")]
impl const MyTrait for Unstable {
    fn func() {}
}

#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_unstable(feature = "unstable2", issue = "none")]
#[const_trait]
pub trait MyTrait2 {
    #[stable(feature = "rust1", since = "1.0.0")]
    fn func2();
}

#[stable(feature = "rust1", since = "1.0.0")]
impl const MyTrait2 for Unstable {
    fn func2() {}
}
