//@ run-pass

use std::ptr;

pub fn main() {
    unsafe {
        let mut x: bool = false;
        // this line breaks it
        ptr::write(&mut x, false);
    }
}
