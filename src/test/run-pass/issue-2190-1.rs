// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::thread::Builder;
use std::thunk::Thunk;

static generations: uint = 1024+256+128+49;

fn spawn(f: Thunk) {
    Builder::new().stack_size(32 * 1024).spawn(move|| f.invoke(())).detach()
}

fn child_no(x: uint) -> Thunk {
    Thunk::new(move|| {
        if x < generations {
            spawn(child_no(x+1));
        }
    })
}

pub fn main() {
    spawn(child_no(0));
}
