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

#![deny(unused_mut)]

// Test that mutating a mutable upvar in a capture-by-value unboxed
// closure does not ice (issue #18238) and marks the upvar as used
// mutably so we do not get a spurious warning about it not needing to
// be declared mutable (issue #18336 and #18769)

fn set(x: &mut usize) { *x = 42; }

fn main() {
    {
        let mut x = 0_usize;
        move || x += 1;
    }
    {
        let mut x = 0_usize;
        move || x += 1;
    }
    {
        let mut x = 0_usize;
        move || set(&mut x);
    }
    {
        let mut x = 0_usize;
        move || set(&mut x);
    }
}
