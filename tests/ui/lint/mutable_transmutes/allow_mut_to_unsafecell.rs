// check-pass

use std::cell::UnsafeCell;
use std::mem::transmute;

fn main() {
    let _a: &mut UnsafeCell<u8> = unsafe { transmute(&mut 0u8) };
    let _a: &UnsafeCell<u8> = unsafe { transmute(&mut 0u8) };
}
