// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(unknown_features)]
#![feature(box_syntax)]
#![feature(old_orphan_check)]

extern crate serialize;

use serialize::{Encodable, Decodable};
use serialize::json;

#[derive(Encodable, Decodable)]
struct A {
    foo: Box<[bool]>,
}

fn main() {
    let obj = A { foo: box [true, false] };
    let s = json::encode(&obj);
    let obj2: A = json::decode(s.as_slice()).unwrap();
    assert!(obj.foo == obj2.foo);
}
