#![feature(rustc_private)]

#[allow(dead_code)]

extern crate serialize as rustc_serialize;

#[derive(RustcDecodable, RustcEncodable,Debug)]
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
