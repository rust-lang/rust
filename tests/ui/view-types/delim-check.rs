//@ build-pass

#![feature(view_types)]
#![allow(irrefutable_let_patterns)]

struct Foo {
    bar: usize,
}

fn main() {
    let foo = Foo { bar: 42 };
    let a = &foo as &Foo.{ bar } else {
        return;
    };
}
