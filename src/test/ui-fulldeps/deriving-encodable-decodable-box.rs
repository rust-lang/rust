// run-pass

#![allow(unused_imports)]
#![feature(rustc_private)]

extern crate rustc_macros;
extern crate rustc_serialize;

use rustc_macros::{Decodable, Encodable};
use rustc_serialize::opaque;
use rustc_serialize::{Decodable, Encodable};

#[derive(Encodable, Decodable)]
struct A {
    foo: Box<[bool]>,
}

fn main() {
    let obj = A { foo: Box::new([true, false]) };
    let mut encoder = opaque::Encoder::new(vec![]);
    obj.encode(&mut encoder).unwrap();
    let mut decoder = opaque::Decoder::new(&encoder.data, 0);
    let obj2 = A::decode(&mut decoder);
    assert_eq!(obj.foo, obj2.foo);
}
