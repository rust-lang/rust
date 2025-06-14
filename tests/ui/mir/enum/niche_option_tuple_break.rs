//@ run-fail
//@ compile-flags: -C debug-assertions
//@ error-pattern: trying to construct an enum from an invalid value 0x3

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
