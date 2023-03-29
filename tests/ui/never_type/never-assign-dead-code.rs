// Test that an assignment of type ! makes the rest of the block dead code.

// check-pass

#![feature(never_type)]
#![allow(drop_copy)]
#![warn(unused)]

fn main() {
    let x: ! = panic!("aah"); //~ WARN unused
    drop(x); //~ WARN unreachable
    //~^ WARN unreachable
}
