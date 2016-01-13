// Copyright 2012-2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regression test for Issue #30530: alloca's created for storing
// intermediate scratch values during brace-less match arms need to be
// initialized with their drop-flag set to "dropped" (or else we end
// up running the destructors on garbage data at the end of the
// function).

pub enum Handler {
    Default,
    #[allow(dead_code)]
    Custom(*mut Box<Fn()>),
}

fn main() {
    take(Handler::Default, Box::new(main));
}

#[inline(never)]
pub fn take(h: Handler, f: Box<Fn()>) -> Box<Fn()> {
    unsafe {
        match h {
            Handler::Custom(ptr) => *Box::from_raw(ptr),
            Handler::Default => f,
        }
    }
}
