// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(fn_traits)]

struct closure_box<'a> {
    cl: Box<FnMut() + 'a>,
}

fn box_it<'r>(x: Box<FnMut() + 'r>) -> closure_box<'r> {
    closure_box {cl: x}
}

fn main() {
    let mut cl_box = {
        let mut i = 3;
        // FIXME (#22405): Replace `Box::new` with `box` here when/if possible.
        box_it(Box::new(|| i += 1)) //~ ERROR `i` does not live long enough
    };
    cl_box.cl.call_mut(());
}
