//@ check-pass

#![allow(clippy::single_match)]

use std::ptr;

fn main() {
    match Some(0_usize) {
        Some(_) => {
            let s = "012345";
            unsafe { ptr::read(s.as_ptr().offset(1) as *const [u8; 5]) };
        },
        _ => (),
    };
}
