//@ run-pass
//@ aux-build: hygiene.rs

#![feature(view_types, view_type_macro, decl_macro)]
#![allow(unused)]

extern crate hygiene;

use std::view::view_type;

pub macro what($name: ident) {
    struct Foo {
        bar: usize,
        $name: usize,
    }

    impl Foo {
        fn f(self: &mut view_type!(Self.{ bar, $name })) {}
    }
}

what!(bar);

struct Bar {
    r#async: (),
}

impl Bar {
    fn f(self: hygiene::view_bar!()) {}
}

fn main() {}
