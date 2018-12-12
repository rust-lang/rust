// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![warn(clippy::redundant_field_names)]
#![allow(unused_variables)]
#![feature(inclusive_range, inclusive_range_fields, inclusive_range_methods)]

#[macro_use]
extern crate derive_new;

use std::ops::{Range, RangeFrom, RangeInclusive, RangeTo, RangeToInclusive};

mod foo {
    pub const BAR: u8 = 0;
}

struct Person {
    gender: u8,
    age: u8,
    name: u8,
    buzz: u64,
    foo: u8,
}

#[derive(new)]
pub struct S {
    v: String,
}

fn main() {
    let gender: u8 = 42;
    let age = 0;
    let fizz: u64 = 0;
    let name: u8 = 0;

    let me = Person {
        gender: gender,
        age: age,

        name,          //should be ok
        buzz: fizz,    //should be ok
        foo: foo::BAR, //should be ok
    };

    // Range expressions
    let (start, end) = (0, 0);

    let _ = start..;
    let _ = ..end;
    let _ = start..end;

    let _ = ..=end;
    let _ = start..=end;

    // Issue #2799
    let _: Vec<_> = (start..end).collect();

    // hand-written Range family structs are linted
    let _ = RangeFrom { start: start };
    let _ = RangeTo { end: end };
    let _ = Range { start: start, end: end };
    let _ = RangeInclusive::new(start, end);
    let _ = RangeToInclusive { end: end };
}
