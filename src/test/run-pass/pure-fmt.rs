// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Testing that calling fmt! (via info!) doesn't complain about impure borrows

struct Big { b: @~str, c: uint, d: int, e: char,
            f: float, g: bool }

fn foo() {
    let a = Big {
        b: @~"hi",
        c: 0,
        d: 1,
        e: 'a',
        f: 0.0,
        g: true
    };
    info!("test %?", a.b);
    info!("test %u", a.c);
    info!("test %i", a.d);
    info!("test %c", a.e);
    info!("test %f", a.f);
    info!("test %b", a.g);
}

pub fn main() {
}
