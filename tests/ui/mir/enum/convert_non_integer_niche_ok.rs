//@ run-pass
//@ compile-flags: -C debug-assertions

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
    let _val: Nested = unsafe { std::mem::transmute::<_, Nested>(Bar { a: 0, b: 0 }) };
    let _val: Nested = unsafe { std::mem::transmute::<_, Nested>(Bar { a: 1, b: 0 }) };
    let _val: Nested = unsafe { std::mem::transmute::<_, Nested>(Bar { a: 2, b: 0 }) };
    let _val: Nested = unsafe { std::mem::transmute::<_, Nested>(Bar { a: 3, b: 0 }) };
}
