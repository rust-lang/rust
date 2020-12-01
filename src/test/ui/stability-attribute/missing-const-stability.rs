#![feature(staged_api)]

#![stable(feature = "rust1", since = "1.0.0")]

#[stable(feature = "foo", since = "1.0.0")]
pub const fn foo() {}
//~^ ERROR rustc_const_stable

#[unstable(feature = "bar", issue = "none")]
pub const fn bar() {} // ok

fn main() {}
