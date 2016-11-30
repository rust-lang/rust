// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that we do some basic error correcton in the tokeniser.

fn main() {
    foo(bar(;
    //~^ NOTE: unclosed delimiter
    //~| NOTE: unclosed delimiter
    //~| ERROR: expected expression, found `;`
    //~| ERROR: unresolved function `foo`
    //~| NOTE: no resolution found
    //~| ERROR: unresolved function `bar`
    //~| NOTE: no resolution found
    //~| ERROR: expected one of `)`, `,`, `.`, `<`, `?`
}
//~^ ERROR: incorrect close delimiter: `}`
//~| ERROR: incorrect close delimiter: `}`
//~| ERROR: expected expression, found `)`
