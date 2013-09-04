// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern mod extra;

use std::task;
use extra::arc::{MutexArc};

fn test_mutex_arc_nested() {
    let arc = ~MutexArc::new(1);
    let arc2 = ~MutexArc::new(*arc);

    do task::spawn || {
        do (*arc2).access |mutex| { //~ ERROR instantiating a type parameter with an incompatible type
        }
    };
}

fn main() {}
