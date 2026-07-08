#![warn(clippy::ptr_as_ptr)]
#![allow(clippy::unnecessary_cast, clippy::unnecessary_operation)]

fn main() {
    let s = std::ptr::null::<()>;
    s() as *const ();
    //~^ ptr_as_ptr
}
