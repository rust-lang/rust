// aux-build:go_trait.rs
// revisions: old re

#![cfg_attr(re, feature(re_rebalance_coherence))]

extern crate go_trait;

use go_trait::{Go,GoMut};
use std::fmt::Debug;
use std::default::Default;

struct MyThingy;

impl Go for MyThingy {
    fn go(&self, arg: isize) { }
}

impl GoMut for MyThingy {
//[old]~^ ERROR conflicting implementations
//[re]~^^ ERROR E0119
    fn go_mut(&mut self, arg: isize) { }
}

fn main() { }
