//@ run-pass
#![allow(dead_code)]

enum E {
    Foo{f: isize},
    Bar,
}

pub fn main() {
    let e = E::Foo{f: 0};
    match e {
        E::Foo{f: 1} => panic!(),
        E::Foo{..} => (),
        _ => panic!(),
    }
}
