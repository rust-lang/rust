// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Testing guarantees provided by once functions.
// This program would segfault if it were legal.

extern mod extra;
use extra::arc;
use std::util;

fn foo(blk: ~once fn()) {
    blk();
    blk(); //~ ERROR use of moved value
}

fn main() {
    let x = arc::ARC(true);
    do foo {
        assert!(*x.get());
        util::ignore(x);
    }
}
