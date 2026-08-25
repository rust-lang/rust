//@ run-pass
//@ compile-flags: -C debug-assertions

#[allow(dead_code)]
enum Foo {
    A,
    B,
}

fn main() {
    let _val: Foo = unsafe { std::mem::transmute::<u8, Foo>(0) };
    let _val: Foo = unsafe { std::mem::transmute::<u8, Foo>(1) };
}
