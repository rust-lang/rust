// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


#![feature(rustc_private)]

extern crate serialize;

use std::io::Cursor;
use std::io::prelude::*;
use std::fmt;
use std::slice;

use serialize::{Encodable, Encoder};
use serialize::json;
use serialize::opaque;

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

fn encode_json<T: Encodable>(val: &T, wr: &mut Cursor<Vec<u8>>) {
    write!(wr, "{}", json::as_json(val));
}
fn encode_opaque<T: Encodable>(val: &T, wr: &mut Cursor<Vec<u8>>) {
    let mut encoder = opaque::Encoder::new(wr);
    val.encode(&mut encoder);
}

pub fn main() {
    let target = Foo{baz: false,};
    let mut wr = Cursor::new(Vec::new());
    let proto = WireProtocol::JSON;
    match proto {
        WireProtocol::JSON => encode_json(&target, &mut wr),
        WireProtocol::Opaque => encode_opaque(&target, &mut wr)
    }
}
