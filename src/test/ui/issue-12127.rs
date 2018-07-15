// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(box_syntax, unboxed_closures)]

fn to_fn_once<A,F:FnOnce<A>>(f: F) -> F { f }
fn do_it(x: &isize) { }

fn main() {
    let x: Box<_> = box 22;
    let f = to_fn_once(move|| do_it(&*x));
    to_fn_once(move|| {
        f();
        f();
        //~^ ERROR: use of moved value: `f`
    })()
}
