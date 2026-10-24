//! Regression test for <https://github.com/rust-lang/rust/issues/83097>.
//!
//! Only the `E0119` conflicting-implementations error used to be reported here, which hid the
//! actual mistake: the unsized field is not the last field of the struct. Both errors are now
//! emitted.

use std::marker::PhantomData;

trait Trait {}

struct Unsized([u8], ());
//~^ ERROR the size for values of type `[u8]` cannot be known at compilation time

struct Foo<T: ?Sized>(PhantomData<T>);

impl<T> Trait for Foo<T> {}
impl Trait for Foo<Unsized> {}
//~^ ERROR conflicting implementations of trait `Trait` for type `Foo<Unsized>`

fn main() {}
