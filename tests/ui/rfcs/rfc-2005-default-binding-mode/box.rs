//@ run-pass
#![allow(unreachable_patterns)]
#![feature(box_patterns)]

struct Foo{}

pub fn main() {
    let b = Box::new(Foo{});
    let box f = &b;
    let _: &Foo = f;

    match &&&b {
        box f => {
            let _: &Foo = f;
        },
        _ => panic!(),
    }
}
