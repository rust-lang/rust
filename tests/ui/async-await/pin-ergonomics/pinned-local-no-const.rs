#![feature(pin_ergonomics)]
#![allow(dead_code, incomplete_features)]

// Makes sure we reject `pin pat`.

struct Foo;

fn foo() {
    let pin x = Foo; //~ ERROR expected one of `:`, `;`, `=`, `@`, or `|`, found `x`
    x = Foo;
}

fn main() {}
