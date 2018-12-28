#![deny(unreachable_code)]

enum Void {}

fn foo(a: (), b: Void) { //~ ERROR functions with parameters of uninhabited types are uncallable
    a
}

trait Foo {
    fn foo(a: Self);

    fn bar(b: Void);
}

impl Foo for Void {
    fn foo(a: Void) {} // ok

    fn bar(b: Void) {} // ok
}

fn main() {}
