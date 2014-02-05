// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn sum(x: &[int]) -> int {
    let mut sum = 0;
    for y in x.iter() { sum += *y; }
    return sum;
}

fn sum_mut(y: &mut [int]) -> int {
    sum(y)
}

fn sum_imm(y: &[int]) -> int {
    sum(y)
}

pub fn main() {}
