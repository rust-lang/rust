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

use std::fmt;
use serialize::{Encodable, Encoder};

pub fn buffer_encode<'a,
                     T:Encodable<serialize::json::Encoder<'a>,fmt::Error>>(
                     to_encode_object: &T)
                     -> String {
    let mut m = String::new();
    {
        let mut encoder =
            serialize::json::Encoder::new(&mut m);
        //~^ ERROR `m` does not live long enough
        to_encode_object.encode(&mut encoder);
    }
    m
}

fn main() {}
