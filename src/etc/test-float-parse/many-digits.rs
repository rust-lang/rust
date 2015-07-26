// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(rand)]

extern crate rand;

mod _common;

use std::char;
use rand::{IsaacRng, Rng, SeedableRng};
use rand::distributions::{Range, Sample};
use _common::{validate, SEED};

fn main() {
    let mut rnd = IsaacRng::from_seed(&SEED);
    let mut range = Range::new(0, 10);
    for _ in 0..5_000_000u64 {
        let num_digits = rnd.gen_range(100, 300);
        let digits = gen_digits(num_digits, &mut range, &mut rnd);
        validate(digits);
    }
}

fn gen_digits<R: Rng>(n: u32, range: &mut Range<u32>, rnd: &mut R) -> String {
    let mut s = String::new();
    for _ in 0..n {
        let digit = char::from_digit(range.sample(rnd), 10).unwrap();
        s.push(digit);
    }
    s
}
