//@ run-pass
//@ compile-flags: -C debug-assertions
// This test depends on the endianess and has a different behavior on
// big endian.
//@ ignore-endian-little

#[allow(dead_code)]
enum Foo {
    A,
    B,
}

#[allow(dead_code)]
struct Bar {
    a: usize,
    b: usize,
}

fn main() {
    let _val: Option<(usize, Foo)> =
        unsafe { std::mem::transmute::<_, Option<(usize, Foo)>>(Bar { a: 3, b: 3 }) };
}
