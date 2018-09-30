// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(unused)]
#![deny(explicit_outlives_requirements)]

use std::fmt::{Debug, Display};

// These examples should live in edition-lint-infer-outlives.rs, but are split
// into this separate file because they can't be `rustfix`'d (and thus, can't
// be part of a `run-rustfix` test file) until rust-lang-nursery/rustfix#141
// is solved

struct TeeOutlivesAyIsDebugBee<'a, 'b, T: 'a + Debug + 'b> {
    //~^ ERROR outlives requirements can be inferred
    tee: &'a &'b T
}

struct TeeWhereOutlivesAyIsDebugBee<'a, 'b, T> where T: 'a + Debug + 'b {
    //~^ ERROR outlives requirements can be inferred
    tee: &'a &'b T
}

struct TeeYooOutlivesAyIsDebugBee<'a, 'b, T, U: 'a + Debug + 'b> {
    //~^ ERROR outlives requirements can be inferred
    tee: T,
    yoo: &'a &'b U
}

struct TeeOutlivesAyYooBeeIsDebug<'a, 'b, T: 'a, U: 'b + Debug> {
    //~^ ERROR outlives requirements can be inferred
    tee: &'a T,
    yoo: &'b U
}

struct TeeOutlivesAyYooIsDebugBee<'a, 'b, T: 'a, U: Debug + 'b> {
    //~^ ERROR outlives requirements can be inferred
    tee: &'a T,
    yoo: &'b U
}

struct TeeOutlivesAyYooWhereBee<'a, 'b, T: 'a, U> where U: 'b {
    //~^ ERROR outlives requirements can be inferred
    tee: &'a T,
    yoo: &'b U
}

struct TeeYooWhereOutlivesAyIsDebugBee<'a, 'b, T, U> where U: 'a + Debug + 'b {
    //~^ ERROR outlives requirements can be inferred
    tee: T,
    yoo: &'a &'b U
}

struct TeeOutlivesAyYooWhereBeeIsDebug<'a, 'b, T: 'a, U> where U: 'b + Debug {
    //~^ ERROR outlives requirements can be inferred
    tee: &'a T,
    yoo: &'b U
}

struct TeeOutlivesAyYooWhereIsDebugBee<'a, 'b, T: 'a, U> where U: Debug + 'b {
    //~^ ERROR outlives requirements can be inferred
    tee: &'a T,
    yoo: &'b U
}

struct TeeWhereOutlivesAyYooWhereBeeIsDebug<'a, 'b, T, U> where T: 'a, U: 'b + Debug {
    //~^ ERROR outlives requirements can be inferred
    tee: &'a T,
    yoo: &'b U
}

struct TeeWhereOutlivesAyYooWhereIsDebugBee<'a, 'b, T, U> where T: 'a, U: Debug + 'b {
    //~^ ERROR outlives requirements can be inferred
    tee: &'a T,
    yoo: &'b U
}

fn main() {}
