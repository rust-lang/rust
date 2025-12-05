#![feature(ptr_metadata, layout_for_ptr)]

use std::{mem, ptr};

trait Foo {}

impl Foo for u32 {}

fn uwu(thin: *const (), meta: &'static ()) -> *const dyn Foo {
    ptr::from_raw_parts(thin, unsafe { mem::transmute(meta) })
}

fn main() {
    unsafe {
        let orig = 1_u32;
        let x = &orig as &dyn Foo;
        let (ptr, meta) = (x as *const dyn Foo).to_raw_parts();
        let ptr = uwu(ptr, mem::transmute(meta));
        let _size = mem::size_of_val_raw(ptr);
    }
}
