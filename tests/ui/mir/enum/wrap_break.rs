//@ run-fail
//@ compile-flags: -C debug-assertions
//@ check-run-results
#![feature(never_type)]
#![allow(invalid_value)]

#[allow(dead_code)]
enum Wrap {
    A(!),
}

fn main() {
    let _val: Wrap = unsafe { std::mem::transmute::<(), Wrap>(()) };
}
