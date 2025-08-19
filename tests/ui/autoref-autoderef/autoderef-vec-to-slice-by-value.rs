//! Tests that a `Vec<isize>` can call a method defined in a trait (`Foo`)
//! implemented for `&[isize]` with a by-value receiver (`self`), relying on auto-dereferencing
//! from `Vec` to `&[isize]` during method resolution.

//@ run-pass

trait Foo {
    fn foo(self);
}

impl<'a> Foo for &'a [isize] {
    fn foo(self) {}
}

pub fn main() {
    let items = vec![ 3, 5, 1, 2, 4 ];
    items.foo();
}
