#![feature(pin_ergonomics)]
#![allow(dead_code, incomplete_features)]

// Makes sure we can handle `pin mut pat` and `pin const pat`.

struct Foo;

fn foo() {
    let pin mut x = Foo;
    let pin const y = Foo;
    x = Foo; // FIXME: this should be an error
    y = Foo; //~ ERROR cannot assign twice to immutable variable `y`

    let (pin mut x, pin const y) = (Foo, Foo);
    x = Foo; // FIXME: this should be an error
    y = Foo; //~ ERROR cannot assign twice to immutable variable `y`
}

fn main() {}
