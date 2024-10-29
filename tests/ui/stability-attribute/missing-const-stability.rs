//@ compile-flags: -Znext-solver
#![feature(staged_api)]
#![feature(const_trait_impl, effects, rustc_attrs, intrinsics)] //~ WARN the feature `effects` is incomplete
#![stable(feature = "stable", since = "1.0.0")]

#[stable(feature = "stable", since = "1.0.0")]
pub const fn foo() {} //~ ERROR function has missing const stability attribute

#[unstable(feature = "unstable", issue = "none")]
pub const fn bar() {} // ok because function is unstable

#[stable(feature = "stable", since = "1.0.0")]
pub struct Foo;
impl Foo {
    #[stable(feature = "stable", since = "1.0.0")]
    pub const fn foo() {} //~ ERROR associated function has missing const stability attribute

    #[unstable(feature = "unstable", issue = "none")]
    pub const fn bar() {} // ok because function is unstable
}

#[stable(feature = "stable", since = "1.0.0")]
#[const_trait]
pub trait Bar {
    #[stable(feature = "stable", since = "1.0.0")]
    fn fun();
}
#[stable(feature = "stable", since = "1.0.0")]
impl const Bar for Foo {
    //~^ ERROR implementation has missing const stability attribute
    fn fun() {}
}

#[stable(feature = "stable", since = "1.0.0")]
#[rustc_intrinsic]
pub const unsafe fn size_of_val<T>(x: *const T) -> usize { 42 }
//~^ ERROR function has missing const stability attribute

extern "rust-intrinsic" {
    #[stable(feature = "stable", since = "1.0.0")]
    #[rustc_const_stable_indirect]
    pub fn min_align_of_val<T>(x: *const T) -> usize;
}

fn main() {}
