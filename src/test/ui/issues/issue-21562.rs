// build-pass (FIXME(62277): could be check-pass?)
#![allow(dead_code)]
#![allow(non_upper_case_globals)]

extern crate core;
use core::marker::Sync;

static SARRAY: [i32; 1] = [11];

struct MyStruct {
    pub arr: *const [i32],
}
unsafe impl Sync for MyStruct {}

static mystruct: MyStruct = MyStruct {
    arr: &SARRAY
};

fn main() {}
