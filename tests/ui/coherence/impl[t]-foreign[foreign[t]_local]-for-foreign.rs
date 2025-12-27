//@ check-pass
//@ compile-flags:--crate-name=test
//@ aux-build:coherence_lib.rs

#![allow(unconstructable_pub_struct)]

extern crate coherence_lib as lib;
use lib::*;
use std::rc::Rc;

struct Local;
impl<T> Remote2<Rc<T>, Local> for usize { }

fn main() {}
