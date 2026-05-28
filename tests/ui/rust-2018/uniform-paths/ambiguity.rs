//@ check-pass
//@ edition:2018

#![allow(non_camel_case_types)]

use std::io; // OK

mod std {
    pub struct io;
}

fn main() {}
