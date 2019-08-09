// build-pass (FIXME(62277): could be check-pass?)
#![allow(dead_code)]
#[derive(Copy,Clone)]
struct Functions {
    a: fn(u32) -> u32,
    b: extern "C" fn(u32) -> u32,
    c: unsafe fn(u32) -> u32,
    d: unsafe extern "C" fn(u32) -> u32
}

pub fn main() {}
