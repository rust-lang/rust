//@ run-pass

#![allow(dead_code)]
struct Foo(());

const FOO: Foo = Foo(match 0 {
    0.. => (),
    _ => (),
});

fn main() {
}
