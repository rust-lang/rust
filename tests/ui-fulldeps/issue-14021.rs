// run-pass

#![allow(unused_mut)]
#![allow(unused_imports)]
#![feature(rustc_private)]

extern crate rustc_macros;
extern crate rustc_serialize;

// Necessary to pull in object code as the rest of the rustc crates are shipped only as rmeta
// files.
#[allow(unused_extern_crates)]
extern crate rustc_driver;

use rustc_macros::{Decodable, Encodable};
use rustc_serialize::opaque::{MemDecoder, MemEncoder};
use rustc_serialize::{Decodable, Encodable, Encoder};

#[derive(Encodable, Decodable, PartialEq, Debug)]
struct UnitLikeStruct;

pub fn main() {
    let obj = UnitLikeStruct;

    let mut encoder = MemEncoder::new();
    obj.encode(&mut encoder);
    let data = encoder.finish();

    let mut decoder = MemDecoder::new(&data, 0);
    let obj2 = UnitLikeStruct::decode(&mut decoder);

    assert_eq!(obj, obj2);
}
