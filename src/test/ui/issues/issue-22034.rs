#![feature(rustc_private)]

extern crate libc;

fn main() {
    let ptr: *mut () = 0 as *mut _;
    let _: &mut dyn Fn() = unsafe {
        &mut *(ptr as *mut dyn Fn())
        //~^ ERROR expected a `std::ops::Fn<()>` closure, found `()`
    };
}
