// check-pass

use std::cell::UnsafeCell;
use std::mem::transmute;

fn main() {
    let _a: &UnsafeCell<u8> = unsafe { transmute(&UnsafeCell::new(0u8)) };
}
