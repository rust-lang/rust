#![allow(unused_variables)]
#![allow(unused_assignments)]
#![allow(dead_code)]
#![deny(unreachable_code)]
#![feature(type_ascription)]

fn a() {
    // the repeat is unreachable:
    let x: [usize; 2] = [return; 2]; //~ ERROR unreachable
}

fn main() { }
