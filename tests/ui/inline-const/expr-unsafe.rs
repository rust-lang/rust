//@ check-pass

#![warn(unused_unsafe)]
#![feature(inline_const)]
const unsafe fn require_unsafe() -> usize { 1 }

fn main() {
    unsafe {
        const {
            require_unsafe();
            unsafe {}
            //~^ WARNING unnecessary `unsafe` block
        }
    }
}
