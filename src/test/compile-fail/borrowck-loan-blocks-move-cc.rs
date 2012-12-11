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
    let mut v = ~3;
    let _w = &mut v; //~ NOTE loan of mutable local variable granted here
    do task::spawn |move v| {
        //~^ ERROR moving out of mutable local variable prohibited due to outstanding loan
        debug!("v=%d", *v);
    }

    let mut v = ~3;
    let _w = &mut v; //~ NOTE loan of mutable local variable granted here
    task::spawn(fn~(move v) {
        //~^ ERROR moving out of mutable local variable prohibited due to outstanding loan
        debug!("v=%d", *v);
    });
}

fn main() {
}
