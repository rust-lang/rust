//@ run-pass
#![allow(dead_code)]

enum Foo {
    Bar,
    Baz,
}

fn foo(f: Foo) {
    match f {
        Foo::Bar => {},
        #[cfg(not(FALSE))]
        Foo::Baz => {},
        #[cfg(FALSE)]
        Basdfwe => {}
    }
}

pub fn main() {}
