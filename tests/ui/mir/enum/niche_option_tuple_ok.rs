//@ run-pass
//@ compile-flags: -C debug-assertions

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
        unsafe { std::mem::transmute::<_, Option<(usize, Foo)>>(Bar { a: 0, b: 0 }) };
    let _val: Option<(usize, Foo)> =
        unsafe { std::mem::transmute::<_, Option<(usize, Foo)>>(Bar { a: 1, b: 0 }) };
}
