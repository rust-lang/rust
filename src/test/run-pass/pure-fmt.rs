// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Testing that calling fmt! (via debug!) doesn't complain about impure borrows

pure fn foo() {
    let a = {
        b: @"hi",
        c: 0,
        d: 1,
        e: 'a',
        f: 0.0,
        g: true
    };
    debug!("test %?", a.b);
    debug!("test %u", a.c);
    debug!("test %i", a.d);
    debug!("test %c", a.e);
    debug!("test %f", a.f);
    debug!("test %b", a.g);
}

fn main() {
}
