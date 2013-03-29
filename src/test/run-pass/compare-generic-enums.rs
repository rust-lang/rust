// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

type an_int = int;

fn cmp(x: Option<an_int>, y: Option<int>) -> bool {
    x == y
}

pub fn main() {
    assert!(!cmp(Some(3), None));
    assert!(!cmp(Some(3), Some(4)));
    assert!(cmp(Some(3), Some(3)));
    assert!(cmp(None, None));
}
