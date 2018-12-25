// run-pass
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]

// aux-build:go_trait.rs

#![feature(specialization)]

extern crate go_trait;

use go_trait::{Go,GoMut};
use std::fmt::Debug;
use std::default::Default;

struct MyThingy;

impl Go for MyThingy {
    fn go(&self, arg: isize) { }
}

impl GoMut for MyThingy {
    fn go_mut(&mut self, arg: isize) { }
}

fn main() { }
