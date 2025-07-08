//@ run-fail
//@ compile-flags: -C debug-assertions
//@ check-run-results

#[allow(dead_code)]
enum Foo {
    A = -2,
    B = -1,
    C = 1,
}

fn main() {
    let _val: Foo = unsafe { std::mem::transmute::<i8, Foo>(-3) };
}
