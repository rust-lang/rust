// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(unused_imports)]
#![allow(unused_must_use)]
// pretty-expanded FIXME #23616

#![feature(rustc_private)]

extern crate rustc_serialize;

use std::fmt;
use rustc_serialize::{Encoder, Encodable};
use rustc_serialize::json;

struct Foo<T: Encodable> {
    v: T,
}

impl<T: Encodable> Drop for Foo<T> {
    fn drop(&mut self) {
        json::encode(&self.v);
    }
}

fn main() {
    let _ = Foo { v: 10 };
}
