// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct Struct { a: uint }

const C: uint = 1;
static S: uint = 1;

const T1: &'static uint = &C;
const T2: &'static uint = &S; //~ ERROR: constants cannot refer to other statics
static T3: &'static uint = &C;
static T4: &'static uint = &S;

const T5: uint = C;
const T6: uint = S; //~ ERROR: constants cannot refer to other statics
//~^ cannot refer to other statics
static T7: uint = C;
static T8: uint = S; //~ ERROR: cannot refer to other statics by value

const T9: Struct = Struct { a: C };
const T10: Struct = Struct { a: S }; //~ ERROR: cannot refer to other statics by value
//~^ ERROR: constants cannot refer to other statics
static T11: Struct = Struct { a: C };
static T12: Struct = Struct { a: S }; //~ ERROR: cannot refer to other statics by value

fn main() {}
