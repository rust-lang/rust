// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(slice_patterns)]
#![allow(unused_variables)]
#![deny(unreachable_patterns)]

fn main() {
    let sl = vec![1,2,3];
    let v: isize = match &*sl {
        &[] => 0,
        &[a,b,c] => 3,
        &[a, ref rest..] => a,
        &[10,a, ref rest..] => 10 //~ ERROR: unreachable pattern
    };
}
