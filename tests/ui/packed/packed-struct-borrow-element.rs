#![allow(dead_code)]
//@ ignore-emscripten weird assertion?

#[repr(packed)]
struct Foo1 {
    bar: u8,
    baz: usize
}

#[repr(packed(2))]
struct Foo2 {
    bar: u8,
    baz: usize
}

#[repr(C, packed(4))]
struct Foo4C {
    bar: u8,
    baz: usize
}

pub fn main() {
    let foo = Foo1 { bar: 1, baz: 2 };
    let brw = &foo.baz; //~ERROR reference to packed field is unaligned
    assert_eq!(*brw, 2);

    let foo = Foo2 { bar: 1, baz: 2 };
    let brw = &foo.baz; //~ERROR reference to packed field is unaligned
    assert_eq!(*brw, 2);
}
