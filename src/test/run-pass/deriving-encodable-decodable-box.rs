// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// pretty-expanded FIXME #23616

#![allow(unknown_features)]
#![feature(box_syntax)]
#![feature(old_orphan_check, rustc_private)]

extern crate serialize;

use serialize::{Encodable, Decodable};
use serialize::json;

#[derive(Encodable, Decodable)]
struct A {
    foo: Box<[bool]>,
}

fn main() {
    // FIXME (#22405): Replace `Box::new` with `box` here when/if possible.
    let obj = A { foo: Box::new([true, false]) };
    let s = json::encode(&obj).unwrap();
    let obj2: A = json::decode(&s).unwrap();
    assert!(obj.foo == obj2.foo);
}
