//! regression test for issue https://github.com/rust-lang/rust/issues/50442
//@ run-pass
#![allow(dead_code)]
enum Void {}

enum Foo {
    A(i32),
    B(Void),
    C(i32),
}

fn main() {
    let _foo = Foo::A(0);
}
