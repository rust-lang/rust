//@ run-pass
#![allow(dead_code)]
// regression test for issue #5625


enum E {
    Foo{f : isize},
    Bar
}

pub fn main() {
    let e = E::Bar;
    match e {
        E::Foo{f: _f} => panic!(),
        _ => (),
    }
}
