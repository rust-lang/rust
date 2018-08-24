#![feature(box_syntax)]
#![feature(rustc_private)]

extern crate serialize;

use serialize::{Encodable, Decodable};
use serialize::json;

#[derive(Encodable, Decodable)]
struct A {
    foo: Box<[bool]>,
}

fn main() {
    let obj = A { foo: Box::new([true, false]) };
    let s = json::encode(&obj).unwrap();
    let obj2: A = json::decode(&s).unwrap();
    assert_eq!(obj.foo, obj2.foo);
}
