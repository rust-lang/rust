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
    assert_eq!((vec!(0i, 1)).to_string(), "[0, 1]".to_string());
    assert_eq!((&[1i, 2]).to_string(), "[1, 2]".to_string());

    let foo = vec!(3i, 4);
    let bar = &[4i, 5];

    assert_eq!(foo.to_string(), "[3, 4]".to_string());
    assert_eq!(bar.to_string(), "[4, 5]".to_string());
}
