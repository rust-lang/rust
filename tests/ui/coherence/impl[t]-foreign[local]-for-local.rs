//@ compile-flags:--crate-name=test
//@ aux-build:coherence_lib.rs
//@ check-pass

#![allow(unconstructable_pub_struct)]

extern crate coherence_lib as lib;
use lib::*;
use std::rc::Rc;

struct Local;

impl Remote1<Local> for Local {}

fn main() {}
