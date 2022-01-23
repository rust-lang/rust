// run-pass

#![allow(unused_imports)]
#![feature(rustc_private)]

extern crate rustc_macros;
extern crate rustc_serialize;

use rustc_macros::{Decodable, Encodable};
use rustc_serialize::json;
use rustc_serialize::{Decodable, Encodable};

#[derive(Encodable, Decodable)]
struct A {
    foo: Box<[bool]>,
}

fn main() {
    let obj = A { foo: Box::new([true, false]) };
    let s = json::encode(&obj).unwrap();
    let obj2: A = json::decode(&s);
    assert_eq!(obj.foo, obj2.foo);
}
