// Test that blanket impl of DispatchFromDyn is rejected.
// regression test for issue <https://github.com/rust-lang/rust/issues/148062>

#![feature(dispatch_from_dyn, unsize)]

use std::marker::Unsize;

impl<T: Unsize<T>> std::ops::DispatchFromDyn<T> for T {}
//~^ ERROR type parameter `T` must be used as the type parameter for some local type (e.g., `MyStruct<T>`)

fn main() {}
