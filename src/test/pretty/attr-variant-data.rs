// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// pp-exact
// Testing that both the inner item and next outer item are
// preserved, and that the first outer item parsed in main is not
// accidentally carried over to each inner function

#![feature(custom_attribute)]
#![feature(custom_derive)]

#[derive(Serialize, Deserialize)]
struct X;

#[derive(Serialize, Deserialize)]
struct WithRef<'a, T: 'a> {
    #[serde(skip_deserializing)]
    t: Option<&'a T>,
    #[serde(serialize_with = "ser_x", deserialize_with = "de_x")]
    x: X,
}

#[derive(Serialize, Deserialize)]
enum EnumWith<T> {
    Unit,
    Newtype(
            #[serde(serialize_with = "ser_x", deserialize_with = "de_x")]
            X),
    Tuple(T,
          #[serde(serialize_with = "ser_x", deserialize_with = "de_x")]
          X),
    Struct {
        t: T,
        #[serde(serialize_with = "ser_x", deserialize_with = "de_x")]
        x: X,
    },
}

#[derive(Serialize, Deserialize)]
struct Tuple<T>(T,
                #[serde(serialize_with = "ser_x", deserialize_with = "de_x")]
                X);

fn main() { }
