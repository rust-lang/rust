// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(unsafe_destructor)]

extern crate serialize;

use std::io::IoError;
use serialize::{Encoder, Encodable};
use serialize::json;

struct Foo<T> {
    v: T,
}

#[unsafe_destructor]
impl<'a, T: Encodable<json::Encoder<'a>, IoError>> Drop for Foo<T> {
    fn drop(&mut self) {
        json::encode(&self.v);
    }
}

fn main() {
    let _ = Foo { v: 10i };
}
