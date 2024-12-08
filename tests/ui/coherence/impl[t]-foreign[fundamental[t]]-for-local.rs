//@ compile-flags:--crate-name=test
//@ aux-build:coherence_lib.rs
//@ check-pass

extern crate coherence_lib as lib;
use lib::*;
use std::rc::Rc;

struct Local;

impl<T> Remote1<Box<T>> for Local {}

impl<'a, T> Remote1<&'a T> for Local {}

fn main() {}
