// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern crate serialize;

use serialize::{Encodable, Encoder};
use serialize::json;
use serialize::ebml::writer;
use std::io::MemWriter;
use std::str::from_utf8_owned;

#[deriving(Encodable)]
struct Foo {
    baz: bool,
}

#[deriving(Encodable)]
struct Bar {
    froboz: uint,
}

enum WireProtocol {
    JSON,
    EBML,
    // ...
}

fn encode_json<'a,
               T: Encodable<json::Encoder<'a>,
                            std::io::IoError>>(val: &T,
                                               wr: &'a mut MemWriter) {
    let mut encoder = json::Encoder::new(wr);
    val.encode(&mut encoder);
}
fn encode_ebml<'a,
               T: Encodable<writer::Encoder<'a, MemWriter>,
                            std::io::IoError>>(val: &T,
                                               wr: &'a mut MemWriter) {
    let mut encoder = writer::Encoder(wr);
    val.encode(&mut encoder);
}

pub fn main() {
    let target = Foo{baz: false,};
    let mut wr = MemWriter::new();
    let proto = JSON;
    match proto {
        JSON => encode_json(&target, &mut wr),
        EBML => encode_ebml(&target, &mut wr)
    }
}
