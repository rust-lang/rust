// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![warn(clippy::serde_api_misuse)]
#![allow(dead_code)]

extern crate serde;

struct A;

impl<'de> serde::de::Visitor<'de> for A {
    type Value = ();

    fn expecting(&self, _: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        unimplemented!()
    }

    fn visit_str<E>(self, _v: &str) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        unimplemented!()
    }

    fn visit_string<E>(self, _v: String) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        unimplemented!()
    }
}

struct B;

impl<'de> serde::de::Visitor<'de> for B {
    type Value = ();

    fn expecting(&self, _: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        unimplemented!()
    }

    fn visit_string<E>(self, _v: String) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        unimplemented!()
    }
}

fn main() {}
