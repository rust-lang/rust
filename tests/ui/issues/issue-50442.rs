//@ run-pass
#![allow(dead_code)]
enum Void {}

enum Foo {
    A(i32),
    B(Void),
    C(i32)
}

fn main() {
    let _foo = Foo::A(0);
}
