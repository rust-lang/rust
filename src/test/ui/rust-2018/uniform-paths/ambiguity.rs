// edition:2018

#![feature(uniform_paths)]

use std::io;
//~^ ERROR `std` import is ambiguous

mod std {
    pub struct io;
}

fn main() {}
