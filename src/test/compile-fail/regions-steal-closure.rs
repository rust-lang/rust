// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(box_syntax)]
#![feature(unboxed_closures)]

struct closure_box<'a> {
    cl: Box<FnMut() + 'a>,
}

fn box_it<'r>(x: Box<FnMut() + 'r>) -> closure_box<'r> {
    closure_box {cl: x}
}

fn main() {
    let cl_box = {
        let mut i = 3is;
        box_it(box || i += 1) //~ ERROR cannot infer
    };
    cl_box.cl.call_mut(());
}
