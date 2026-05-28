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

fn main() {
    let _val: Nested = unsafe { std::mem::transmute::<u32, Nested>(0) };
    let _val: Nested = unsafe { std::mem::transmute::<u32, Nested>(1) };
    let _val: Nested = unsafe { std::mem::transmute::<u32, Nested>(2) };
    let _val: Nested = unsafe { std::mem::transmute::<u32, Nested>(3) };
}
