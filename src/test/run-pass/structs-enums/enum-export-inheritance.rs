// run-pass
#![allow(dead_code)]
// pretty-expanded FIXME #23616

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
