// build-pass (FIXME(62277): could be check-pass?)
#![allow(dead_code)]
#[repr(u64)]
enum A {
    A = 0u64,
    B = !0u64,
}

fn cmp() -> A {
    A::B
}

fn main() {}
