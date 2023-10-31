// build-pass
// compile-flags:-Zmir-opt-level=4 --emit=mir

#![feature(allocator_api)]

use std::alloc::{Allocator, Global, Layout};

fn main() {
    let layout: Layout = None.unwrap();
    let ptr: *mut u8 = Global.allocate(layout).unwrap().as_ptr() as _;
}
