// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

mod a {
    pub mod b {
        pub mod c {
            pub struct S;
            pub struct Z;
        }
        pub struct W;
    }
}

macro_rules! import {
    (1 $p: path) => (use $p;);
    (2 $p: path) => (use $p::{Z};);
    (3 $p: path) => (use $p::*;);
}

import! { 1 a::b::c::S }
import! { 2 a::b::c }
import! { 3 a::b }

fn main() {
    let s = S;
    let z = Z;
    let w = W;
}
