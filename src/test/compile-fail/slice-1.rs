// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test slicing expr[..] is an error and gives a helpful error message.

struct Foo;

fn main() {
    let x = Foo;
    x[..]; //~ ERROR incorrect slicing expression: `[..]`
    //~^ NOTE use `expr[]` to construct a slice of the whole of expr
}
