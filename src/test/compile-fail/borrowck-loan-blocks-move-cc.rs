// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn borrow(v: &int, f: fn(x: &int)) {
    f(v);
}

fn box_imm() {
    let v = ~3;
    let _w = &v; //~ NOTE loan of immutable local variable granted here
    do task::spawn {
        debug!("v=%d", *v);
        //~^ ERROR by-move capture of immutable local variable prohibited due to outstanding loan
    }

    let v = ~3;
    let _w = &v; //~ NOTE loan of immutable local variable granted here
    task::spawn(fn~() {
        debug!("v=%d", *v);
        //~^ ERROR by-move capture of immutable local variable prohibited due to outstanding loan
    });
}

fn main() {
}
