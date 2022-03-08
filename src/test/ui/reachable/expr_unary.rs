#![feature(never_type)]
#![allow(unused_variables)]
#![allow(unused_assignments)]
#![allow(dead_code)]
#![deny(unreachable_code)]

fn foo() {
    let x: ! = * { return; }; //~ ERROR unreachable
    //~| ERROR type `!` cannot be dereferenced
}

fn main() { }
