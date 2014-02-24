// ignore-fast
// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-fast

extern crate serialize;

use serialize::{json, Decodable};

trait JD : Decodable<json::Decoder> { }

fn exec<T: JD>() {
    let doc = json::from_str("").unwrap();
    let mut decoder = json::Decoder::new(doc);
    let _v: T = Decodable::decode(&mut decoder);
    fail!()
}

pub fn main() {}
