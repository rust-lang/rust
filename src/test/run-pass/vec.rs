// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.




// -*- rust -*-
pub fn main() {
    let v: ~[int] = ~[10, 20];
    assert!((v[0] == 10));
    assert!((v[1] == 20));
    let mut x: int = 0;
    assert!((v[x] == 10));
    assert!((v[x + 1] == 20));
    x = x + 1;
    assert!((v[x] == 20));
    assert!((v[x - 1] == 10));
}
