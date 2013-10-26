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
    let mut x: int = 1;
    x *= 2;
    info!("{}", x);
    assert_eq!(x, 2);
    x += 3;
    info!("{}", x);
    assert_eq!(x, 5);
    x *= x;
    info!("{}", x);
    assert_eq!(x, 25);
    x /= 5;
    info!("{}", x);
    assert_eq!(x, 5);
}
