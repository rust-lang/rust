#![allow(non_snake_case)]
#![allow(dead_code)]
#![allow(unused_variables)]
// Test that the lifetime of rvalues in for loops is extended
// to the for loop itself.

use std::ops::Drop;

static mut FLAGS: u64 = 0;

struct Box<T> { f: T }
struct AddFlags { bits: u64 }

fn AddFlags(bits: u64) -> AddFlags {
    AddFlags { bits: bits }
}

fn arg(exp: u64, _x: &AddFlags) {
    check_flags(exp);
}

fn pass<T>(v: T) -> T {
    v
}

fn check_flags(exp: u64) {
    unsafe {
        let x = FLAGS;
        FLAGS = 0;
        println!("flags {}, expected {}", x, exp);
        assert_eq!(x, exp);
    }
}

impl AddFlags {
    fn check_flags(&self, exp: u64) -> &AddFlags {
        check_flags(exp);
        self
    }

    fn bits(&self) -> u64 {
        self.bits
    }
}

impl Drop for AddFlags {
    fn drop(&mut self) {
        unsafe {
            FLAGS = FLAGS + self.bits;
        }
    }
}

pub fn main() {
    // The array containing [AddFlags] should not be dropped until
    // after the for loop:
    for x in &[AddFlags(1)] {
        check_flags(0);
    }
    check_flags(1);
}
