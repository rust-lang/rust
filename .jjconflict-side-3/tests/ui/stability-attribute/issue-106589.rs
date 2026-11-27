// #![feature(staged_api)] // note: `staged_api` not enabled

#![stable(feature = "foo", since = "1.0.0")]
//~^ ERROR stability attributes may not be used outside of the standard library

#[unstable(feature = "foo", issue = "none")]
//~^ ERROR stability attributes may not be used outside of the standard library
fn foo_unstable() {}

fn main() {}
