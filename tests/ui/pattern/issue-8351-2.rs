//@ run-pass
#![allow(dead_code)]

enum E {
    Foo{f: isize, b: bool},
    Bar,
}

pub fn main() {
    let e = E::Foo{f: 0, b: false};
    match e {
        E::Foo{f: 1, b: true} => panic!(),
        E::Foo{b: false, f: 0} => (),
        _ => panic!(),
    }
}
