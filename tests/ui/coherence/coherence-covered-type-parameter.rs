//@ run-pass
#![allow(dead_code)]
//@ aux-build:coherence_lib.rs


extern crate coherence_lib as lib;
use lib::Remote;

struct Foo<T>(T);

impl<T> Remote for Foo<T> { }

fn main() { }
