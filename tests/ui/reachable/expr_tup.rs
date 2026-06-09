#![allow(unused_variables)]
#![allow(unused_assignments)]
#![allow(dead_code)]
#![deny(unreachable_code)]
#![feature(type_ascription)]

fn a() {
    // the `2` is unreachable:
    let x: (usize, usize) = (return, 2); //~ ERROR unreachable
}

fn b() {
    // the tuple is unreachable:
    let x: (usize, usize) = (2, return); //~ ERROR unreachable
}

fn main() { }
