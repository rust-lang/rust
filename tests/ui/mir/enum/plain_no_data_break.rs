//@ run-fail
//@ compile-flags: -C debug-assertions
//@ check-run-results

#[repr(u32)]
#[allow(dead_code)]
enum Foo {
    A = 2,
    B,
}

fn main() {
    let _val: Foo = unsafe { std::mem::transmute::<u32, Foo>(1) };
}
