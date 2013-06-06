// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-test
extern mod extra;

use hashmap;
use extra::json;
use extra::serialization::{Deserializable, deserialize};

trait JD : Deserializable<json::Deserializer> { }
//type JD = Deserializable<json::Deserializer>;

fn exec<T:JD>() {
    let doc = result::unwrap(json::from_str(""));
    let _v: T = deserialize(&json::Deserializer(doc));
    fail!()
}

pub fn main() {}
