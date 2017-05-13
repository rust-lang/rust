// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// pretty-expanded FIXME #23616

#![allow(unknown_features)]

// Test that `Fn(isize) -> isize + 'static` parses as `(Fn(isize) -> isize) +
// 'static` and not `Fn(isize) -> (isize + 'static)`. The latter would
// cause a compilation error. Issue #18772.

fn adder(y: isize) -> Box<Fn(isize) -> isize + 'static> {
    // FIXME (#22405): Replace `Box::new` with `box` here when/if possible.
    Box::new(move |x| y + x)
}

fn main() {}
