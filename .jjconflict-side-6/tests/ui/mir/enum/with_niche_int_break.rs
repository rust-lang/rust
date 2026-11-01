//@ run-crash
//@ compile-flags: -C debug-assertions
//@ error-pattern: trying to construct an enum from an invalid value

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

fn main() {
    let _val: Nested = unsafe { std::mem::transmute::<u32, Nested>(u32::MAX) };
}
