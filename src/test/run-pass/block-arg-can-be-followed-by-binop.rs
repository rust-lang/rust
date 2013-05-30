// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub fn main() {
    let v = ~[-1f, 0f, 1f, 2f, 3f];

    // Trailing expressions don't require parentheses:
    let y = do vec::foldl(0f, v) |x, y| { x + *y } + 10f;

    assert_eq!(y, 15f);
}
