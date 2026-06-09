//@ run-crash
//@ compile-flags: -C debug-assertions
//@ error-pattern: trying to construct an enum from an invalid value 0x0
#![feature(never_type)]
#![allow(invalid_value)]

#[allow(dead_code)]
enum Wrap {
    A(!),
}

fn main() {
    let _val: Wrap = unsafe { std::mem::transmute::<(), Wrap>(()) };
}
