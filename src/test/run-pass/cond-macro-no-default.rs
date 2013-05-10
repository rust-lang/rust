// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn clamp<T:Copy + Ord + Signed>(x: T, mn: T, mx: T) -> T {
    cond!(
        (x > mx) { return mx; }
        (x < mn) { return mn; }
    )
    return x;
}

fn main() {
    assert_eq!(clamp(1, 2, 4), 2);
    assert_eq!(clamp(8, 2, 4), 4);
    assert_eq!(clamp(3, 2, 4), 3);
}
