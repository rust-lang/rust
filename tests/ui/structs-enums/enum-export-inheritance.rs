//@ run-pass
#![allow(dead_code)]

mod a {
    pub enum Foo {
        Bar,
        Baz,
        Boo
    }
}

pub fn main() {
    let _x = a::Foo::Bar;
}
