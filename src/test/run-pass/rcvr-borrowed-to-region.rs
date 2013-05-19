// xfail-test
// xfail'd due to segfaults with by-value self.

// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait get {
    fn get(self) -> int;
}

// Note: impl on a slice
impl get for &'self int {
    fn get(self) -> int {
        return **self;
    }
}

pub fn main() {
    /*
    let x = @mut 6;
    let y = x.get();
    assert_eq!(y, 6);
    */

    let x = @6;
    let y = x.get();
    debug!("y=%d", y);
    assert_eq!(y, 6);

    let mut x = ~6;
    let y = x.get();
    debug!("y=%d", y);
    assert_eq!(y, 6);

    let x = ~6;
    let y = x.get();
    debug!("y=%d", y);
    assert_eq!(y, 6);

    let x = &6;
    let y = x.get();
    debug!("y=%d", y);
    assert_eq!(y, 6);
}
