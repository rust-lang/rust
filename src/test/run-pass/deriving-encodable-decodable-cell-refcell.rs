// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This briefuly tests the capability of `Cell` and `RefCell` to implement the
// `Encodable` and `Decodable` traits via `#[deriving(Encodable, Decodable)]`

extern crate serialize;

use std::cell::{Cell, RefCell};
use std::io::MemWriter;
use serialize::{Encodable, Decodable};
use serialize::json;

#[deriving(Encodable, Decodable)]
struct A {
    baz: int
}

#[deriving(Encodable, Decodable)]
struct B {
    foo: Cell<bool>,
    bar: RefCell<A>,
}

fn main() {
    let obj = B {
        foo: Cell::new(true),
        bar: RefCell::new( A { baz: 2 } )
    };
    let s = json::encode(&obj);
    let obj2: B = json::decode(s.as_slice()).unwrap();
    assert!(obj.foo.get() == obj2.foo.get());
    assert!(obj.bar.borrow().baz == obj2.bar.borrow().baz);
}
