// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub fn main() {
    assert_eq!((vec!(0, 1)).to_str(), ~"[0, 1]");
    assert_eq!((&[1, 2]).to_str(), ~"[1, 2]");

    let foo = vec!(3, 4);
    let bar = &[4, 5];

    assert_eq!(foo.to_str(), ~"[3, 4]");
    assert_eq!(bar.to_str(), ~"[4, 5]");
}
