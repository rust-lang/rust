// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(warnings)]

// this should compile in a reasonable amount of time
fn rust_type_id(name: &str) {
    if "bool" == &name[..]
        || "uint" == &name[..]
        || "u8" == &name[..]
        || "u16" == &name[..]
        || "u32" == &name[..]
        || "f32" == &name[..]
        || "f64" == &name[..]
        || "i8" == &name[..]
        || "i16" == &name[..]
        || "i32" == &name[..]
        || "i64" == &name[..]
        || "Self" == &name[..]
        || "str" == &name[..]
    {
        unreachable!();
    }
}

fn main() {}
