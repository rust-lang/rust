// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[feature(managed_boxes)];

// Tests for if as expressions returning boxed types
fn test_box() {
    let rs = if true { @100 } else { @101 };
    assert_eq!(*rs, 100);
}

fn test_str() {
    let rs = if true { ~"happy" } else { ~"sad" };
    assert_eq!(rs, ~"happy");
}

pub fn main() { test_box(); test_str(); }
