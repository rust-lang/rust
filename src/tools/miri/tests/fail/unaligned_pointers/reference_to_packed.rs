// This should fail even without validation/SB
//@compile-flags: -Zmiri-disable-validation -Zmiri-disable-stacked-borrows

#![allow(dead_code, unused_variables)]

use std::{ptr, mem};

#[repr(packed)]
struct Foo {
    x: i32,
    y: i32,
}

unsafe fn raw_to_ref<'a, T>(x: *const T) -> &'a T {
    mem::transmute(x)
}

fn main() {
    // Try many times as this might work by chance.
    for _ in 0..20 {
        let foo = Foo { x: 42, y: 99 };
        // There seem to be implicit reborrows, which make the error already appear here
        let p: &i32 = unsafe { raw_to_ref(ptr::addr_of!(foo.x)) }; //~ERROR: alignment 4 is required
        let i = *p;
    }
}
