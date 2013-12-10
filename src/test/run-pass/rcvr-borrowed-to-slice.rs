// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait sum {
    fn sum_(self) -> int;
}

// Note: impl on a slice
impl<'a> sum for &'a [int] {
    fn sum_(self) -> int {
        self.iter().fold(0, |a, &b| a + b)
    }
}

fn call_sum(x: &[int]) -> int { x.sum_() }

pub fn main() {
    let x = ~[1, 2, 3];
    let y = call_sum(x);
    info!("y=={}", y);
    assert_eq!(y, 6);

    let x = ~[1, 2, 3];
    let y = x.sum_();
    info!("y=={}", y);
    assert_eq!(y, 6);

    let x = ~[1, 2, 3];
    let y = x.sum_();
    info!("y=={}", y);
    assert_eq!(y, 6);
}
