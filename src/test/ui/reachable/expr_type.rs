#![allow(unused_variables)]
#![allow(unused_assignments)]
#![allow(dead_code)]
#![deny(unreachable_code)]
#![feature(type_ascription)]

fn a() {
    // the cast is unreachable:
    let x = {return}: !; //~ ERROR unreachable
}

fn main() { }
