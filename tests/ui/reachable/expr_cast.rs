#![allow(unused_variables)]
#![allow(unused_assignments)]
#![allow(dead_code)]
#![deny(unreachable_code)]
#![feature(never_type, type_ascription)]

fn a() {
    // the cast is unreachable:
    let x = {return} as !; //~ ERROR unreachable
    //~| ERROR non-primitive cast
}

fn main() { }
