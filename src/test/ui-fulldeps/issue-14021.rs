// run-pass

#![allow(unused_mut)]
#![allow(unused_imports)]
#![feature(rustc_private)]

extern crate rustc_macros;
extern crate rustc_serialize;

use rustc_macros::{Decodable, Encodable};
use rustc_serialize::opaque::{MemDecoder, MemEncoder};
use rustc_serialize::{Decodable, Encodable, Encoder};

#[derive(Encodable, Decodable, PartialEq, Debug)]
struct UnitLikeStruct;

pub fn main() {
    let obj = UnitLikeStruct;

    let mut encoder = MemEncoder::new();
    obj.encode(&mut encoder);
    let data = encoder.finish().unwrap();

    let mut decoder = MemDecoder::new(&data, 0);
    let obj2 = UnitLikeStruct::decode(&mut decoder);

    assert_eq!(obj, obj2);
}
