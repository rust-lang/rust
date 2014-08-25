// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct Foo<'a,'b> {
    x: &'a int,
    y: &'b int,
}

impl<'a,'b> Foo<'a,'b> {
    // The number of errors is related to the way invariance works.
    fn bar(self: Foo<'b,'a>) {}
    //~^ ERROR mismatched types: expected `Foo<'a,'b>`, found `Foo<'b,'a>`
    //~^^ ERROR mismatched types: expected `Foo<'a,'b>`, found `Foo<'b,'a>`
    //~^^^ ERROR mismatched types: expected `Foo<'b,'a>`, found `Foo<'a,'b>`
    //~^^^^ ERROR mismatched types: expected `Foo<'b,'a>`, found `Foo<'a,'b>`
}

fn main() {}

