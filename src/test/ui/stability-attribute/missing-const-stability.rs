#![feature(staged_api)]
#![feature(const_trait_impl)]
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

// FIXME Once #![feature(const_trait_impl)] is allowed to be stable, add a test
// for const trait impls. Right now, a "trait methods cannot be stable const fn"
// error is emitted. This occurs prior to the lint being tested here, such that
// the lint cannot currently be tested on this use case.

fn main() {}
