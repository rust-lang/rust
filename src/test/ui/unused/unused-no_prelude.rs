#![no_implicit_prelude]
#![feature(no_prelude)]
#![deny(unused_attributes)]
#![no_prelude]
//~^ ERROR: unused attribute

mod unused {
    #![no_prelude]
    //~^ ERROR: unused attribute
}

fn main() {}
