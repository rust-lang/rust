//@ run-fail
//@ compile-flags: -C debug-assertions
//@ error-pattern: trying to construct an enum from an invalid value 0x3

#[allow(dead_code)]
enum Foo {
    A,
    B,
}

fn main() {
    let _val: Foo = unsafe { std::mem::transmute::<u8, Foo>(3) };
}
