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
