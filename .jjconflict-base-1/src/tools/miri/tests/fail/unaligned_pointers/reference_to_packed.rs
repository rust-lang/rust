// This should fail even without SB
//@compile-flags: -Zmiri-disable-stacked-borrows -Cdebug-assertions=no

#![allow(dead_code, unused_variables)]

use std::{mem, ptr};

#[repr(packed)]
struct Foo {
    x: i32,
    y: i32,
}

unsafe fn raw_to_ref<'a, T>(x: *const T) -> &'a T {
    mem::transmute(x) //~ERROR: required 4 byte alignment
}

fn main() {
    // Try many times as this might work by chance.
    for _ in 0..20 {
        let foo = Foo { x: 42, y: 99 };
        let p: &i32 = unsafe { raw_to_ref(ptr::addr_of!(foo.x)) };
        let i = *p;
    }
}
