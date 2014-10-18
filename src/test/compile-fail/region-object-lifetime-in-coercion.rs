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

fn a(v: &[u8]) -> Box<Clone + 'static> {
    let x: Box<Clone + 'static> = box v; //~ ERROR does not outlive
    x
}

fn b(v: &[u8]) -> Box<Clone + 'static> {
    box v //~ ERROR does not outlive
}

fn c(v: &[u8]) -> Box<Clone> {
    box v // OK thanks to lifetime elision
}

fn d<'a,'b>(v: &'a [u8]) -> Box<Clone+'b> {
    box v //~ ERROR does not outlive
}

fn e<'a:'b,'b>(v: &'a [u8]) -> Box<Clone+'b> {
    box v // OK, thanks to 'a:'b
}

fn main() { }
