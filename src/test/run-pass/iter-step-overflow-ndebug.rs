// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -C debug_assertions=no

fn main() {
    let mut it = u8::max_value()..;
    assert_eq!(it.next().unwrap(), 255);
    assert_eq!(it.next().unwrap(), u8::min_value());

    let mut it = i8::max_value()..;
    assert_eq!(it.next().unwrap(), 127);
    assert_eq!(it.next().unwrap(), i8::min_value());
}
