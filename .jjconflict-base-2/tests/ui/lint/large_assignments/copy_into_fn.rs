//@ build-fail

#![feature(large_assignments)]
#![move_size_limit = "1000"]
#![deny(large_assignments)]
#![allow(unused)]

// We want copy semantics, because moving data into functions generally do not
// translate to actual `memcpy`s.
#[derive(Copy, Clone)]
struct Data([u8; 9999]);

fn main() {
    one_arg(Data([0; 9999])); //~ ERROR large_assignments

    // each individual large arg shall have its own span
    many_args(Data([0; 9999]), true, Data([0; 9999]));
    //~^ ERROR large_assignments
    //~| ERROR large_assignments
}

fn one_arg(a: Data) {}

fn many_args(a: Data, b: bool, c: Data) {}
