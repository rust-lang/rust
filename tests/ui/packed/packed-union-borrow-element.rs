#![allow(dead_code)]
//@ ignore-emscripten weird assertion?

#[repr(packed)]
#[derive(Clone, Copy)]
struct Foo1(usize);

#[repr(packed(4))]
#[derive(Clone, Copy)]
struct Foo4(usize);

#[repr(packed(2))]
union Bar2 {
    foo1: Foo1,
    foo4: Foo4,
}

pub fn main() {
    let bar = Bar2 { foo1: Foo1(2) };
    let brw = unsafe { &bar.foo1.0 }; //~ERROR reference to field of packed struct is unaligned
    assert_eq!(*brw, 2);

    let bar = Bar2 { foo4: Foo4(2) };
    let brw = unsafe { &bar.foo4.0 }; //~ERROR reference to field of packed union is unaligned
    assert_eq!(*brw, 2);
}
