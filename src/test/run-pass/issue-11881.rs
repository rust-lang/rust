// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern crate rbml;
extern crate serialize;

use std::io;
use std::io::{IoError, IoResult, SeekStyle};
use std::slice;

use serialize::{Encodable, Encoder};
use serialize::json;

use rbml::writer;
use rbml::io::SeekableMemWriter;

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
    RBML,
    // ...
}

fn encode_json<'a,
               T: Encodable<json::Encoder<'a>,
                            std::io::IoError>>(val: &T,
                                               wr: &'a mut SeekableMemWriter) {
    let mut encoder = json::Encoder::new(wr);
    val.encode(&mut encoder);
}
fn encode_rbml<'a,
               T: Encodable<writer::Encoder<'a, SeekableMemWriter>,
                            std::io::IoError>>(val: &T,
                                               wr: &'a mut SeekableMemWriter) {
    let mut encoder = writer::Encoder::new(wr);
    val.encode(&mut encoder);
}

pub fn main() {
    let target = Foo{baz: false,};
    let mut wr = SeekableMemWriter::new();
    let proto = JSON;
    match proto {
        JSON => encode_json(&target, &mut wr),
        RBML => encode_rbml(&target, &mut wr)
    }
}
