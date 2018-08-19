// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
//
// Regression test for #51351 and #52133: In the case of #51351,
// late-bound regions (like 'a) that were unused within the arguments of
// a function were overlooked and could case an ICE. In the case of #52133,
// LBR defined on the creator function needed to be added to the free regions
// of the closure, as they were not present in the closure's generic
// declarations otherwise.
//
// compile-pass

#![feature(nll)]

fn creash<'a>() {
    let x: &'a () = &();
}

fn produce<'a>() {
   move || {
        let x: &'a () = &();
   };
}

fn main() {}
