// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that we are able to compile calls to associated fns like
// `decode()` where the bound on the `Self` parameter references a
// lifetime parameter of the trait. This example indicates why trait
// lifetime parameters must be early bound in the type of the
// associated item.

use std::marker;

pub enum Value<'v> {
    A(&'v str),
    B,
}

pub trait Decoder<'v> {
    fn read(&mut self) -> Value<'v>;
}

pub trait Decodable<'v, D: Decoder<'v>>
    : marker::PhantomFn<(), &'v int>
{
    fn decode(d: &mut D) -> Self;
}

impl<'v, D: Decoder<'v>> Decodable<'v, D> for () {
    fn decode(d: &mut D) -> () {
        match d.read() {
            Value::A(..) => (),
            Value::B => Decodable::decode(d),
        }
    }
}

pub fn main() { }
