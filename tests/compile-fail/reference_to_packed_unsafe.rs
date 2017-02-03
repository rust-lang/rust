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
    let p: *const i32 = &foo.x;
    let x = unsafe { *p + foo.x }; //~ ERROR tried to access memory with alignment 1, but alignment 4 is required
}
