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

struct an_enum<'self>(&'self int);
struct a_class<'self> { x:&'self int }

fn a_fn1<'a,'b>(e: an_enum<'a>) -> an_enum<'b> {
    return e; //~ ERROR mismatched types: expected `an_enum/&'b ` but found `an_enum/&'a `
}

fn a_fn3<'a,'b>(e: a_class<'a>) -> a_class<'b> {
    return e; //~ ERROR mismatched types: expected `a_class/&'b ` but found `a_class/&'a `
}

fn a_fn4<'a,'b>() {
    let _: int<'a> = 1; //~ ERROR region parameters are not allowed on this type
}

fn main() { }
