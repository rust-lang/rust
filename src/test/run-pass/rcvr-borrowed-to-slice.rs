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
    fn sum() -> int;
}

// Note: impl on a slice
impl &[int]: sum {
    fn sum() -> int {
        let mut sum = 0;
        for vec::each(self) |e| { sum += *e; }
        return sum;
    }
}

fn call_sum(x: &[int]) -> int { x.sum() }

fn main() {
    let x = ~[1, 2, 3];
    let y = call_sum(x);
    debug!("y==%d", y);
    assert y == 6;

    let mut x = ~[1, 2, 3];
    let y = x.sum();
    debug!("y==%d", y);
    assert y == 6;

    let x = ~[1, 2, 3];
    let y = x.sum();
    debug!("y==%d", y);
    assert y == 6;
}
