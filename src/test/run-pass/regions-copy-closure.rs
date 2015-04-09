// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


#![allow(unknown_features)]
#![feature(unboxed_closures, core)]

struct closure_box<'a> {
    cl: Box<FnMut() + 'a>,
}

fn box_it<'a>(x: Box<FnMut() + 'a>) -> closure_box<'a> {
    closure_box {cl: x}
}

pub fn main() {
    let mut i = 3;
    assert_eq!(i, 3);
    {
        let cl = || i += 1;
        // FIXME (#22405): Replace `Box::new` with `box` here when/if possible.
        let mut cl_box = box_it(Box::new(cl));
        cl_box.cl.call_mut(());
    }
    assert_eq!(i, 4);
}
