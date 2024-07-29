//@ check-pass

#![allow(unsafe_cell_transmutes)]

use std::cell::UnsafeCell;

fn main() {
    let _: &UnsafeCell<u8> = unsafe { &*(&0u8 as *const u8 as *const UnsafeCell<u8>) };
}
