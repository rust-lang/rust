// run-pass

#![allow(unused_must_use)]
#![allow(dead_code)]
#![allow(unused_imports)]
#![feature(rustc_private)]

extern crate rustc_macros;
extern crate rustc_serialize;

use std::fmt;
use std::io::prelude::*;
use std::io::Cursor;
use std::slice;

use rustc_macros::Encodable;
use rustc_serialize::json;
use rustc_serialize::opaque;
use rustc_serialize::{Encodable, Encoder};

#[derive(Encodable)]
struct Foo {
    baz: bool,
}

#[derive(Encodable)]
struct Bar {
    froboz: usize,
}

enum WireProtocol {
    JSON,
    Opaque,
    // ...
}

fn encode_json<T: for<'a> Encodable<json::Encoder<'a>>>(val: &T, wr: &mut Cursor<Vec<u8>>) {
    write!(wr, "{}", json::as_json(val));
}
fn encode_opaque<T: Encodable<opaque::Encoder>>(val: &T, wr: Vec<u8>) {
    let mut encoder = opaque::Encoder::new(wr);
    val.encode(&mut encoder);
}

pub fn main() {
    let target = Foo { baz: false };
    let proto = WireProtocol::JSON;
    match proto {
        WireProtocol::JSON => encode_json(&target, &mut Cursor::new(Vec::new())),
        WireProtocol::Opaque => encode_opaque(&target, Vec::new()),
    }
}
