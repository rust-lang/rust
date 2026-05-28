//@ compile-flags: -Zwrite-long-types-to-disk=yes
use std::ops::Deref;

// Make sure that method probe error reporting doesn't get too tangled up
// on this infinite deref impl. See #130224.

struct Wrap<T>(T);
impl<T> Deref for Wrap<T> {
    type Target = Wrap<Wrap<T>>;
    fn deref(&self) -> &Wrap<Wrap<T>> { todo!() }
}

fn main() {
    Wrap(1).lmao();
    //~^ ERROR reached the recursion limit
    //~| ERROR no method named `lmao`
}
