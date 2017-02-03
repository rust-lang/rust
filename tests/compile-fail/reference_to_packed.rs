#![allow(dead_code, unused_variables)]

#[repr(packed)]
struct Foo {
    x: i32,
    y: i32,
}

fn main() {
    let foo = Foo {
        x: 42,
        y: 99,
    };
    let p = &foo.x;
    let i = *p; //~ ERROR tried to access memory with alignment 1, but alignment 4 is required
}