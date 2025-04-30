//@compile-flags: -Zmiri-tree-borrows
#![feature(box_as_ptr)]
#[path = "../../utils/mod.rs"]
#[macro_use]
mod utils;

use std::cell::UnsafeCell;

pub fn main() {
    let cell = UnsafeCell::new(42);
    let mut root = Box::new(cell);

    let a = Box::as_mut_ptr(&mut root);
    unsafe {
        name!(a);
        let alloc_id = alloc_id!(a);
        print_state!(alloc_id);
    }
}
