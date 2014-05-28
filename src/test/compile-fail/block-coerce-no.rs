// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Make sure that fn-to-block coercion isn't incorrectly lifted over
// other tycons.

extern crate debug;

fn coerce(b: ||) -> extern fn() {
    fn lol(f: extern fn(v: ||) -> extern fn(),
           g: ||) -> extern fn() { return f(g); }
    fn fn_id(f: extern fn()) -> extern fn() { return f }
    return lol(fn_id, b);
    //~^ ERROR mismatched types
}

fn main() {
    let i = 8;
    let f = coerce(|| println!("{:?}", i) );
    f();
}
