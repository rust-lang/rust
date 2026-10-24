//@ compile-flags:--crate-name=test
//@ aux-build:coherence_lib.rs
//@ check-pass

extern crate coherence_lib as lib;
use lib::*;
use std::rc::Rc;

struct Local<T>(Rc<T>);

impl Remote1<Local<i32>> for i32 {}
impl<T> Remote1<Local<T>> for f32 {}

fn main() {}
