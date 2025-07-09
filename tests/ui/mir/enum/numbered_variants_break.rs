//@ run-fail
//@ compile-flags: -C debug-assertions
//@ check-run-results

#[allow(dead_code)]
enum Foo {
    A,
    B,
}

fn main() {
    let _val: Foo = unsafe { std::mem::transmute::<u8, Foo>(3) };
}
