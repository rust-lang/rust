#![deny(unreachable_code)]

enum Void {}

mod hide {
    pub struct PrivatelyUninhabited(::Void);

    pub struct PubliclyUninhabited(pub ::Void);
}

// Check that functions with (publicly) uninhabited parameters trigger a lint.

fn foo(a: (), b: Void) { //~ ERROR functions with parameters of uninhabited types are uncallable
    a
}

fn bar(a: (), b: hide::PrivatelyUninhabited) { // ok
    a
}

fn baz(a: (), b: hide::PubliclyUninhabited) {
    //~^ ERROR functions with parameters of uninhabited types are uncallable
    a
}

// Check that trait methods with uninhabited parameters do not trigger a lint
// (at least for now).

trait Foo {
    fn foo(a: Self);

    fn bar(b: Void);
}

impl Foo for Void {
    fn foo(a: Void) {} // ok

    fn bar(b: Void) {} // ok
}

fn main() {}
