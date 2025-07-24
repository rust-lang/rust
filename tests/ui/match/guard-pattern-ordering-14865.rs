//! Regression test for https://github.com/rust-lang/rust/issues/14865

//@ run-pass
#![allow(dead_code)]

enum X {
    Foo(usize),
    Bar(bool)
}

fn main() {
    let x = match X::Foo(42) {
        X::Foo(..) => 1,
        _ if true => 0,
        X::Bar(..) => panic!("Oh dear")
    };
    assert_eq!(x, 1);

    let x = match X::Foo(42) {
        _ if true => 0,
        X::Foo(..) => 1,
        X::Bar(..) => panic!("Oh dear")
    };
    assert_eq!(x, 0);
}
