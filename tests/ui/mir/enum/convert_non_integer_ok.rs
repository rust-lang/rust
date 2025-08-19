//@ run-pass
//@ compile-flags: -C debug-assertions

#[allow(dead_code)]
#[repr(u32)]
enum Foo {
    A,
    B,
}

#[allow(dead_code)]
struct Bar {
    a: u32,
}

fn main() {
    let _val: Foo = unsafe { std::mem::transmute::<_, Foo>(Bar { a: 0 }) };
    let _val: Foo = unsafe { std::mem::transmute::<_, Foo>(Bar { a: 1 }) };
}
