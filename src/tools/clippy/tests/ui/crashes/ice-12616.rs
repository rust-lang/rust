#![warn(clippy::ptr_as_ptr)]
#![allow(clippy::unnecessary_operation, clippy::unnecessary_cast)]

fn main() {
    let s = std::ptr::null::<()>;
    s() as *const ();
    //~^ ptr_as_ptr
}
