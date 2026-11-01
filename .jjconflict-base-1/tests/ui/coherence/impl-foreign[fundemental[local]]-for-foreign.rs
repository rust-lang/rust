//@ compile-flags:--crate-name=test
//@ aux-build:coherence_lib.rs
//@ check-pass

extern crate coherence_lib as lib;
use lib::*;
use std::rc::Rc;

struct Local;
struct Local1<T>(Rc<T>);

impl Remote1<Box<Local>> for i32 {}
impl Remote1<Box<Local1<i32>>> for f64 {}
impl<T> Remote1<Box<Local1<T>>> for f32 {}

fn main() {}
