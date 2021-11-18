#![feature(staged_api)]
#![stable(feature = "stable", since = "1.0.0")]

#[stable(feature = "stable", since = "1.0.0")]
pub const fn foo() {} //~ ERROR function has missing const stability attribute

#[unstable(feature = "unstable", issue = "none")]
pub const fn bar() {} // ok for now

#[stable(feature = "stable", since = "1.0.0")]
pub struct Foo;
impl Foo {
    #[stable(feature = "stable", since = "1.0.0")]
    pub const fn foo() {} //~ ERROR associated function has missing const stability attribute

    #[unstable(feature = "unstable", issue = "none")]
    pub const fn bar() {} // ok for now
}

// FIXME When #![feature(const_trait_impl)] is stabilized, add tests for const
// trait impls. Right now, a "trait methods cannot be stable const fn" error is
// emitted, but that's not in the scope of this test.

fn main() {}
