// pretty-expanded FIXME #23616

use std::ptr;

pub fn main() {
    unsafe {
        let mut x: bool = false;
        // this line breaks it
        ptr::write(&mut x, false);
    }
}
