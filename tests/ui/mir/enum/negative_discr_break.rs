//@ run-crash
//@ compile-flags: -C debug-assertions
//@ error-pattern: trying to construct an enum from an invalid value 0xfd

#[allow(dead_code)]
enum Foo {
    A = -2,
    B = -1,
    C = 1,
}

fn main() {
    let _val: Foo = unsafe { std::mem::transmute::<i8, Foo>(-3) };
}
