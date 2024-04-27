//@ run-pass
#![allow(dead_code)]

#[derive(PartialEq, Eq)]
struct Foo {
    x: u32
}

const FOO: Foo = Foo { x: 0 };

fn main() {
    let y = Foo { x: 1 };
    match y {
        FOO => { }
        _ => { }
    }
}
