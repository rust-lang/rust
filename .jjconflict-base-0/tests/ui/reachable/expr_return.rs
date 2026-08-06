#![allow(unused_variables)]
#![allow(unused_assignments)]
#![allow(dead_code)]
#![deny(unreachable_code)]
#![feature(type_ascription)]

fn a() {
    // Here we issue that the "2nd-innermost" return is unreachable,
    // but we stop there.
    let x: () = {return {return {return;}}}; //~ ERROR unreachable
}

fn main() { }
