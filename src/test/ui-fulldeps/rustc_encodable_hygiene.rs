// run-pass

#![feature(rustc_private)]

extern crate rustc_macros;
#[allow(dead_code)]
extern crate rustc_serialize;

use rustc_macros::{Decodable, Encodable};

#[derive(Decodable, Encodable, Debug)]
struct A {
    a: String,
}

trait Trait {
    fn encode(&self);
}

impl<T> Trait for T {
    fn encode(&self) {
        unimplemented!()
    }
}

fn main() {}
