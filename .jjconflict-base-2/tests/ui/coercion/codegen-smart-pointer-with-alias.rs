//@ build-pass

// Regression test for <https://github.com/rust-lang/rust/issues/139812>.

// Make sure that the unsize coercion we collect in mono for `Signal<i32> -> Signal<dyn Any>`
// doesn't choke on the fact that the inner unsized field of `Signal<T>` is a (trivial) alias.
// This exercises a normalize call that is necessary since we're getting a type from the type
// system, which isn't guaranteed to be normalized after substitution.

#![feature(coerce_unsized)]

use std::ops::CoerceUnsized;

trait Mirror {
    type Assoc: ?Sized;
}
impl<T: ?Sized> Mirror for T {
    type Assoc = T;
}

trait Any {}
impl<T> Any for T {}

struct Signal<'a, T: ?Sized>(<&'a T as Mirror>::Assoc);

// This `CoerceUnsized` impl isn't special; it's a bit more restricted than we'd see in the wild,
// but this ICE also reproduces if we were to make it general over `Signal<T> -> Signal<U>`.
impl<'a> CoerceUnsized<Signal<'a, dyn Any>> for Signal<'a, i32> {}

fn main() {
    Signal(&1i32) as Signal<dyn Any>;
}
