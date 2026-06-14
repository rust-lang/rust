#![feature(view_types)]
#![allow(irrefutable_let_patterns)]

struct Foo {
    bar: usize,
}

fn main() {
    let foo = Foo { bar: 42 };
    let a = &foo as &Foo.{ bar } else {
        //~^ ERROR right curly brace `}` before `else` in a `let...else` statement not allowed
        return;
    };
}
