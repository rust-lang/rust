//@ run-fail
//@ compile-flags: -C debug-assertions
//@ error-pattern: trying to construct an enum from an invalid value 0x5

#[allow(dead_code)]
#[repr(u16)]
enum Mix {
    A,
    B(u16),
}

#[allow(dead_code)]
enum Nested {
    C(Mix),
    D,
    E,
}

#[allow(dead_code)]
struct Bar {
    a: u16,
    b: u16,
}

fn main() {
    let _val: Nested = unsafe { std::mem::transmute::<_, Nested>(Bar { a: 5, b: 0 }) };
}
