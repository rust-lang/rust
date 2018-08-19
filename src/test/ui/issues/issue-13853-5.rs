// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait Deserializer<'a> { }

trait Deserializable {
    fn deserialize_token<'a, D: Deserializer<'a>>(_: D, _: &'a str) -> Self;
}

impl<'a, T: Deserializable> Deserializable for &'a str {
    //~^ ERROR type parameter `T` is not constrained
    fn deserialize_token<D: Deserializer<'a>>(_x: D, _y: &'a str) -> &'a str {
    }
}

fn main() {}
