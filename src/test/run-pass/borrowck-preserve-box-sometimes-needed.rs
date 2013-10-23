// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// exec-env:RUST_POISON_ON_FREE=1

#[feature(managed_boxes)];

fn switcher(x: Option<@int>) {
    let mut x = x;
    match x {
        Some(@y) => { y.clone(); x = None; }
        None => { }
    }
    assert_eq!(x, None);
}

pub fn main() {
    switcher(None);
    switcher(Some(@3));
}
