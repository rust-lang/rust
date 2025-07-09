//@ run-fail
//@ compile-flags: -C debug-assertions
//@ check-run-results

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
        unsafe { std::mem::transmute::<_, Option<(usize, Foo)>>(Bar { a: 3, b: 3 }) };
}
