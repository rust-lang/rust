#![feature(core_intrinsics, volatile)]

use std::intrinsics::{
    unaligned_volatile_load, unaligned_volatile_store, volatile_load, volatile_store,
};
use std::ptr::{read_volatile, write_volatile};

pub fn main() {
    unsafe {
        let mut i: isize = 1;
        volatile_store(&mut i, 2);
        assert_eq!(volatile_load(&i), 2);
    }
    unsafe {
        let mut i: isize = 1;
        unaligned_volatile_store(&mut i, 2);
        assert_eq!(unaligned_volatile_load(&i), 2);
    }
    unsafe {
        let mut i: isize = 1;
        write_volatile(&mut i, 2);
        assert_eq!(read_volatile(&i), 2);
    }
}
