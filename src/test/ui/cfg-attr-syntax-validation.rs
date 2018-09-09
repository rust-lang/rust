// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[cfg] //~ ERROR `cfg` is not followed by parentheses
struct S1;

#[cfg = 10] //~ ERROR `cfg` is not followed by parentheses
struct S2;

#[cfg()] //~ ERROR `cfg` predicate is not specified
struct S3;

#[cfg(a, b)] //~ ERROR multiple `cfg` predicates are specified
struct S4;

#[cfg("str")] //~ ERROR `cfg` predicate key cannot be a literal
struct S5;

#[cfg(a::b)] //~ ERROR `cfg` predicate key must be an identifier
struct S6;

#[cfg(a())] //~ ERROR invalid predicate `a`
struct S7;

#[cfg(a = 10)] //~ ERROR literal in `cfg` predicate value must be a string
struct S8;

macro_rules! generate_s9 {
    ($expr: expr) => {
        #[cfg(feature = $expr)] //~ ERROR `cfg` is not a well-formed meta-item
        struct S9;
    }
}

generate_s9!(concat!("nonexistent"));
