#![feature(rustc_private)]

extern crate libc;

fn main() {
    let ptr: *mut () = 0 as *mut _;
    let _: &mut Fn() = unsafe {
        &mut *(ptr as *mut Fn())
        //~^ ERROR expected a `std::ops::Fn<()>` closure, found `()`
    };
}
