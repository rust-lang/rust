// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that we don't ICE due to encountering unsubstituted type
// parameters when untupling FnOnce parameters during translation of
// an unboxing shim.

#![feature(unboxed_closures)]

fn main() {
    let _: Box<FnOnce<(),()>> = box move |&mut:| {};
}
