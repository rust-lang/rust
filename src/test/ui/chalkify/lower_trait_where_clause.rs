// check-pass
// compile-flags: -Z chalk

use std::borrow::Borrow;

trait Foo<'a, 'b, T, U>
where
    T: Borrow<U> + ?Sized,
    U: ?Sized + 'b,
    'a: 'b,
    Box<T>:, // NOTE(#53696) this checks an empty list of bounds.
{
}

fn main() {
}
