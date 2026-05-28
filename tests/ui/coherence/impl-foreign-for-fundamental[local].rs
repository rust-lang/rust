//@ compile-flags:--crate-name=test
//@ aux-build:coherence_lib.rs
//@ check-pass

extern crate coherence_lib as lib;
use lib::*;
use std::rc::Rc;

struct Local;
struct Local1<T>(Rc<T>);

impl Remote for Box<Local> {}
impl<T> Remote for Box<Local1<T>> {}

fn main() {}
