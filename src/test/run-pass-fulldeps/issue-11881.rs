#![allow(unused_must_use)]
#![allow(dead_code)]
#![allow(unused_imports)]

#![feature(rustc_private)]

extern crate serialize;
use serialize as rustc_serialize;

use std::io::Cursor;
use std::io::prelude::*;
use std::fmt;
use std::slice;

use serialize::{Encodable, Encoder};
use serialize::json;
use serialize::opaque;

#[derive(RustcEncodable)]
struct Foo {
    baz: bool,
}

#[derive(RustcEncodable)]
struct Bar {
    froboz: usize,
}

enum WireProtocol {
    JSON,
    Opaque,
    // ...
}

fn encode_json<T: Encodable>(val: &T, wr: &mut Cursor<Vec<u8>>) {
    write!(wr, "{}", json::as_json(val));
}
fn encode_opaque<T: Encodable>(val: &T, wr: Vec<u8>) {
    let mut encoder = opaque::Encoder::new(wr);
    val.encode(&mut encoder);
}

pub fn main() {
    let target = Foo{baz: false,};
    let proto = WireProtocol::JSON;
    match proto {
        WireProtocol::JSON => encode_json(&target, &mut Cursor::new(Vec::new())),
        WireProtocol::Opaque => encode_opaque(&target, Vec::new())
    }
}
