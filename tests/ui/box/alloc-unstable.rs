// run-pass
#![feature(allocator_api)]

use std::boxed::Box;

fn main() {
    let _boxed: Box<u32, _> = Box::new(10);
}
