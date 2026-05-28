//@ run-crash
//@ compile-flags: -C debug-assertions
//@ error-pattern: trying to construct an enum from an invalid value 0x1

#[repr(u32)]
#[allow(dead_code)]
enum Foo {
    A = 2,
    B,
}

fn main() {
    let _val: Foo = unsafe { std::mem::transmute::<u32, Foo>(1) };
}
