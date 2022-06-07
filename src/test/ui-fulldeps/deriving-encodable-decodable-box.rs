// run-pass

#![allow(unused_imports)]
#![feature(rustc_private)]

extern crate rustc_macros;
extern crate rustc_serialize;

use rustc_macros::{Decodable, Encodable};
use rustc_serialize::opaque;
use rustc_serialize::{Decodable, Encodable, Encoder};

#[derive(Encodable, Decodable)]
struct A {
    foo: Box<[bool]>,
}

fn main() {
    let obj = A { foo: Box::new([true, false]) };

    let mut encoder = opaque::Encoder::new();
    obj.encode(&mut encoder);
    let data = encoder.finish();

    let mut decoder = opaque::Decoder::new(&data, 0);
    let obj2 = A::decode(&mut decoder);

    assert_eq!(obj.foo, obj2.foo);
}
