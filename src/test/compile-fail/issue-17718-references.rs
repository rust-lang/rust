// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct Struct { a: usize }

const C: usize = 1;
static S: usize = 1;

const T1: &'static usize = &C;
const T2: &'static usize = &S; //~ ERROR: constants cannot refer to statics
static T3: &'static usize = &C;
static T4: &'static usize = &S;

const T5: usize = C;
const T6: usize = S; //~ ERROR: constants cannot refer to statics
//~^ cannot refer to statics
static T7: usize = C;
static T8: usize = S; //~ ERROR: cannot refer to other statics by value

const T9: Struct = Struct { a: C };
const T10: Struct = Struct { a: S }; //~ ERROR: cannot refer to statics by value
//~^ ERROR: constants cannot refer to statics
static T11: Struct = Struct { a: C };
static T12: Struct = Struct { a: S }; //~ ERROR: cannot refer to other statics by value

fn main() {}
