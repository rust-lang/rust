// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Check that explicit region bounds are allowed on the various
// nominal types (but not on other types) and that they are type
// checked.

struct TupleStruct<'a>(&'a isize);
struct Struct<'a> { x:&'a isize }

fn a_fn1<'a,'b>(e: TupleStruct<'a>) -> TupleStruct<'b> {
    return e; //~ ERROR mismatched types
}

fn a_fn3<'a,'b>(e: Struct<'a>) -> Struct<'b> {
    return e; //~ ERROR mismatched types
}

fn main() { }
