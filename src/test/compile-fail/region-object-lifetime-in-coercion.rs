// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that attempts to implicitly coerce a value into an
// object respect the lifetime bound on the object type.

trait Foo {}
impl<'a> Foo for &'a [u8] {}

fn a(v: &[u8]) -> Box<Foo + 'static> {
    let x: Box<Foo + 'static> = Box::new(v);
    //~^ ERROR explicit lifetime required in the type of `v` [E0621]
    x
}

fn b(v: &[u8]) -> Box<Foo + 'static> {
    Box::new(v)
        //~^ ERROR explicit lifetime required in the type of `v` [E0621]
}

fn c(v: &[u8]) -> Box<Foo> {
    // same as previous case due to RFC 599

    Box::new(v)
        //~^ ERROR explicit lifetime required in the type of `v` [E0621]
}

fn d<'a,'b>(v: &'a [u8]) -> Box<Foo+'b> {
    Box::new(v)
        //~^ ERROR cannot infer an appropriate lifetime due to conflicting
}

fn e<'a:'b,'b>(v: &'a [u8]) -> Box<Foo+'b> {
    Box::new(v) // OK, thanks to 'a:'b
}

fn main() { }
