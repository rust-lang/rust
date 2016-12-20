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
    x: &'a isize,
    y: &'b isize,
}

impl<'a,'b> Foo<'a,'b> {
    fn bar(self:
           Foo<'b,'a>
    //~^ ERROR mismatched method receiver
    //~| expected type `Foo<'a, 'b>`
    //~| found type `Foo<'b, 'a>`
    //~| lifetime mismatch
    //~| ERROR mismatched method receiver
    //~| expected type `Foo<'a, 'b>`
    //~| found type `Foo<'b, 'a>`
    //~| lifetime mismatch
           ) {}
}

fn main() {}
