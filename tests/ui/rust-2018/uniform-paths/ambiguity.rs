// edition:2018

#![allow(non_camel_case_types)]

use std::io;
//~^ ERROR `std` is ambiguous

mod std {
    pub struct io;
}

fn main() {}
