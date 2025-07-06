//@ run-fail
//@ compile-flags: -C debug-assertions
// This test depends on the endianess and has a different behavior on
// little endian.
//@ ignore-endian-little
//@ error-pattern: trying to construct an enum from an invalid value

#[allow(dead_code)]
#[repr(u32)]
enum Foo {
    A,
    B,
}

#[allow(dead_code)]
struct Bar {
    a: u16,
    b: u16,
}

fn main() {
    let _val: Foo = unsafe { std::mem::transmute::<_, Foo>(Bar { a: 1, b: 0 }) };
}
