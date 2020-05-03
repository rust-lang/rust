#![feature(unsafe_block_in_unsafe_fn)]
#![warn(unsafe_op_in_unsafe_fn)]
#![deny(unused_unsafe)]

unsafe fn unsf() {}

unsafe fn foo() {
    unsf();
    //~^ WARNING call to unsafe function is unsafe and requires unsafe block
}

unsafe fn bar() {
    unsafe { unsf() } // no error
}

unsafe fn baz() {
    unsafe { unsafe { unsf() } }
    //~^ ERROR unnecessary `unsafe` block
}

#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn qux() {
    unsf(); // no error
}

fn main() {}
