// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// pretty-expanded FIXME #23616

#![allow(unknown_features)]
#![feature(box_syntax)]

// This is a regression test that the metadata for the
// name_pool::methods impl in the other crate is reachable from this
// crate.

// aux-build:crate-method-reexport-grrrrrrr2.rs

extern crate crate_method_reexport_grrrrrrr2;

pub fn main() {
    use crate_method_reexport_grrrrrrr2::rust::add;
    use crate_method_reexport_grrrrrrr2::rust::cx;
    let x: Box<_> = box ();
    x.cx();
    let y = ();
    y.add("hi".to_string());
}
