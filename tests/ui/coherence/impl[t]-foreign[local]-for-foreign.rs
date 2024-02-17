//@ compile-flags:--crate-name=test
//@ aux-build:coherence_lib.rs
//@ check-pass

extern crate coherence_lib as lib;
use lib::*;
use std::rc::Rc;

struct Local;

impl<T> Remote1<Local> for Rc<T> {}
impl<T> Remote1<Local> for Vec<Box<T>> {}

fn main() {}
