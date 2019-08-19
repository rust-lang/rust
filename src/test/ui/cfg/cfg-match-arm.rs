// run-pass
#![allow(dead_code)]
// pretty-expanded FIXME #23616

enum Foo {
    Bar,
    Baz,
}

fn foo(f: Foo) {
    match f {
        Foo::Bar => {},
        #[cfg(not(asdfa))]
        Foo::Baz => {},
        #[cfg(afsd)]
        Basdfwe => {}
    }
}

pub fn main() {}
