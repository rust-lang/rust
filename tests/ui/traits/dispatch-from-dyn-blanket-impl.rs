// Test that blanket impl of DispatchFromDyn is rejected.
// regression test for issue <https://github.com/rust-lang/rust/issues/148062>

#![feature(dispatch_from_dyn)]

impl<T> std::ops::DispatchFromDyn<T> for T {}
//~^ ERROR type parameter `T` must be used as an argument to some local type (e.g., `MyStruct<T>`)
//~| ERROR the trait `DispatchFromDyn` may only be implemented for a coercion between structures

fn main() {}
