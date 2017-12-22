// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Various tests related to testing how region inference works
// with respect to the object receivers.

trait Foo {
    fn borrowed<'a>(&'a self) -> &'a ();
}

// Borrowed receiver but two distinct lifetimes, we get an error.
fn borrowed_receiver_different_lifetimes<'a,'b>(x: &'a Foo) -> &'b () {
    x.borrowed() //~ ERROR cannot infer
}

fn main() {}
