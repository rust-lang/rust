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

// FIXME (#22405): Replace `Box::new` with `box` here when/if possible.

fn a(v: &[u8]) -> Box<Foo + 'static> {
    //~^ the lifetime cannot outlive the anonymous lifetime #1 defined on the block
    let x: Box<Foo + 'static> = Box::new(v);
    //~^ ERROR cannot infer an appropriate lifetime due to conflicting
    //~| ERROR cannot infer an appropriate lifetime due to conflicting
    //~| NOTE cannot infer an appropriate lifetime
    //~| NOTE so that expression is assignable (expected &[u8], found &[u8])
    //~| NOTE so that the type `&[u8]` will meet its required lifetime bounds
    //~| NOTE the lifetime must be valid for the static lifetime
    //~| NOTE the lifetime must be valid for the static lifetime
    x
}

fn b(v: &[u8]) -> Box<Foo + 'static> {
    //~^ the lifetime cannot outlive the anonymous lifetime #1 defined on the block
    Box::new(v)
        //~^ ERROR cannot infer an appropriate lifetime due to conflicting
        //~| ERROR cannot infer an appropriate lifetime due to conflicting
        //~| NOTE cannot infer an appropriate lifetime
        //~| NOTE so that expression is assignable (expected &[u8], found &[u8])
        //~| NOTE so that the type `&[u8]` will meet its required lifetime bounds
        //~| NOTE the lifetime must be valid for the static lifetime
        //~| NOTE the lifetime must be valid for the static lifetime
}

fn c(v: &[u8]) -> Box<Foo> {
    //~^ the lifetime cannot outlive the anonymous lifetime #1 defined on the block
    // same as previous case due to RFC 599

    Box::new(v)
        //~^ ERROR cannot infer an appropriate lifetime due to conflicting
        //~| ERROR cannot infer an appropriate lifetime due to conflicting
        //~| NOTE cannot infer an appropriate lifetime
        //~| NOTE so that expression is assignable (expected &[u8], found &[u8])
        //~| NOTE so that the type `&[u8]` will meet its required lifetime bounds
        //~| NOTE the lifetime must be valid for the static lifetime
        //~| NOTE the lifetime must be valid for the static lifetime
}

fn d<'a,'b>(v: &'a [u8]) -> Box<Foo+'b> {
    Box::new(v)
        //~^ ERROR cannot infer an appropriate lifetime due to conflicting
        //~| NOTE cannot infer an appropriate lifetime
}

fn e<'a:'b,'b>(v: &'a [u8]) -> Box<Foo+'b> {
    Box::new(v) // OK, thanks to 'a:'b
}

fn main() { }
