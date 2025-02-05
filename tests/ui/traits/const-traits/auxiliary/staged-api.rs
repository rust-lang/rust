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

    #[stable(feature = "rust1", since = "1.0.0")]
    fn default<T: ~const MyTrait>() {
        // This call is const-unstable, but permitted here since we inherit
        // the `rustc_const_unstable` above.
        T::func();
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
pub struct Unstable;

#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_unstable(feature = "unstable", issue = "none")]
impl const MyTrait for Unstable {
    fn func() {}

    fn default<T: ~const MyTrait>() {
        // This call is const-unstable, but permitted here since we inherit
        // the `rustc_const_unstable` above.
        T::func();
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
pub struct Unstable2;

#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_unstable(feature = "unstable2", issue = "none")]
impl const MyTrait for Unstable2 {
    fn func() {}
}
